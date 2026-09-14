"""
PyTorch port of ../js/model.js.

This is not "inspired by" the JS model — it is a deliberate formula-for-formula
port, so that weights trained here can be exported (see bridge.py) and loaded
directly into the tf.js model in the browser and produce the same outputs.
Every op below was cross-checked against a live tf.js instance before this
file was written (conv 'same' padding for stride 1 and for our stride-2/
kernel-4 downsample, LayerNormalization's epsilon=1e-3 and biased variance,
UpSampling2D's nearest-neighbor behavior, and the sinusoidal time-embedding
formula) — see bridge_test/ for that check. If you change H/W/kernel sizes,
same_pad() below reproduces TF's SAME padding generally, not just for 20x10.

Layer names (tDense1, stemConv, bottleneck[i].conv1, ...) intentionally match
js/model.js's `layers` object one-for-one — bridge.py relies on this to build
the export JSON, and js/model.js's loadWeights() relies on the same names to
apply it back.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

X0_CLIP = 1.5
SELF_COND_PROB = 0.5


def same_pad(in_size, kernel, stride):
    """TF/Keras 'SAME' padding amount for one spatial dim. Verified against a
    live tf.js layer for (in=10,k=4,s=2)->(1,1) and (in=20,k=4,s=2)->(1,1);
    for stride=1 with an odd kernel this reduces to the usual (k-1)//2 each side."""
    out = math.ceil(in_size / stride)
    pad_total = max((out - 1) * stride + kernel - in_size, 0)
    before = pad_total // 2
    after = pad_total - before
    return before, after


class ConvSame2d(nn.Module):
    """Conv2d operating on NHWC tensors (to mirror model.js 1:1) with TF-style
    SAME padding, computed via explicit F.pad + a stride/padding=0 conv2d
    (torch has no built-in SAME for stride>1)."""

    def __init__(self, in_ch, out_ch, kernel, stride=1):
        super().__init__()
        self.kernel, self.stride = kernel, stride
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=0)

    def forward(self, x_nhwc):
        x = x_nhwc.permute(0, 3, 1, 2)  # -> NCHW
        _, _, h, w = x.shape
        ph0, ph1 = same_pad(h, self.kernel, self.stride)
        pw0, pw1 = same_pad(w, self.kernel, self.stride)
        x = F.pad(x, (pw0, pw1, ph0, ph1))
        x = self.conv(x)
        return x.permute(0, 2, 3, 1)  # -> NHWC


class ChannelLayerNorm(nn.Module):
    """Matches tf.keras.layers.LayerNormalization(axis=-1): per-pixel
    normalization over the channel vector, biased variance, eps=1e-3
    (Keras's default — NOT PyTorch LayerNorm's 1e-5 default)."""

    def __init__(self, channels, eps=1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):  # x: [N,H,W,C]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class FiLM(nn.Module):
    def __init__(self, temb_dim, channels):
        super().__init__()
        self.proj = nn.Linear(temb_dim, channels * 2)
        self.channels = channels

    def forward(self, h, temb):  # h: [N,H,W,C], temb: [N,temb_dim]
        f = self.proj(temb).view(-1, 1, 1, self.channels * 2)
        scale, shift = f[..., : self.channels], f[..., self.channels :]
        return h * (scale + 1) + shift


def sinusoidal_time_embedding(t_norm, dim):
    """t_norm: [N] float tensor in [0,1]. Matches model.js's timeEmbeddingBatch exactly."""
    half = max(dim // 2, 1)
    i = torch.arange(half, device=t_norm.device, dtype=t_norm.dtype)
    freqs = torch.exp(-math.log(10000.0) * i / max(half - 1, 1))
    args = torch.outer(t_norm, freqs) * (2 * math.pi)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    elif emb.shape[1] > dim:
        emb = emb[:, :dim]
    return emb


class BottleneckBlock(nn.Module):
    def __init__(self, hidden, kernel, temb_dim):
        super().__init__()
        self.conv1 = ConvSame2d(hidden, hidden, kernel)
        self.norm1 = ChannelLayerNorm(hidden)
        self.film1 = FiLM(temb_dim, hidden)
        self.conv2 = ConvSame2d(hidden, hidden, kernel)
        self.norm2 = ChannelLayerNorm(hidden)
        self.film2 = FiLM(temb_dim, hidden)

    def forward(self, h, temb):
        b = F.relu(self.film1(self.norm1(self.conv1(h)), temb))
        b = self.film2(self.norm2(self.conv2(b)), temb)  # no activation, matches convBlock(..., activate=False)
        return F.relu(h + b)


class DiffusionModel(nn.Module):
    """height/width/vocab_size fixed at construction (must match the dataset).
    hidden/time_dim/bottleneck_blocks/bottleneck_kernel are the knobs for
    'much bigger' — see train.py --help for sane ranges."""

    def __init__(self, height, width, vocab_size, hidden=128, time_dim=64,
                 bottleneck_blocks=8, bottleneck_kernel=7):
        super().__init__()
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f"height and width must both be even (got {width}x{height}) - the stride-2 "
                f"downsample + nearest-2x upsample only round-trips back to the exact original "
                f"size when both are even; an odd dimension would otherwise fail with a shape "
                f"mismatch deep inside forward() instead of this clear error."
            )
        self.height, self.width, self.vocab_size = height, width, vocab_size
        self.hidden, self.time_dim = hidden, time_dim

        self.tDense1 = nn.Linear(time_dim, 128)
        self.tDense2 = nn.Linear(128, hidden)

        self.stemConv = ConvSame2d(vocab_size * 2, hidden, 3)
        self.stemNorm = ChannelLayerNorm(hidden)
        self.stemFilm = FiLM(hidden, hidden)

        self.downConv = ConvSame2d(hidden, hidden, 4, stride=2)
        self.downNorm = ChannelLayerNorm(hidden)
        self.downFilm = FiLM(hidden, hidden)

        self.bottleneck = nn.ModuleList(
            [BottleneckBlock(hidden, bottleneck_kernel, hidden) for _ in range(bottleneck_blocks)]
        )

        self.refine1Conv = ConvSame2d(hidden * 2, hidden, 3)
        self.refine1Norm = ChannelLayerNorm(hidden)
        self.refine1Film = FiLM(hidden, hidden)

        self.refine2Conv = ConvSame2d(hidden, hidden, 3)
        self.refine2Norm = ChannelLayerNorm(hidden)
        self.refine2Film = FiLM(hidden, hidden)

        self.convOut = ConvSame2d(hidden, vocab_size, 3)

    def forward(self, x, self_cond, t_norm):
        """x, self_cond: [N,H,W,vocab] analog-bits tensors. t_norm: [N] in [0,1]."""
        temb = F.relu(self.tDense1(sinusoidal_time_embedding(t_norm, self.time_dim)))
        temb = F.relu(self.tDense2(temb))

        xin = torch.cat([x, self_cond], dim=-1)

        skip = F.relu(self.stemFilm(self.stemNorm(self.stemConv(xin)), temb))
        h = F.relu(self.downFilm(self.downNorm(self.downConv(skip)), temb))

        for block in self.bottleneck:
            h = block(h, temb)

        h = h.permute(0, 3, 1, 2)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = h.permute(0, 2, 3, 1)
        h = torch.cat([h, skip], dim=-1)

        h = F.relu(self.refine1Film(self.refine1Norm(self.refine1Conv(h)), temb))
        r2 = self.refine2Film(self.refine2Norm(self.refine2Conv(h)), temb)
        h = F.relu(h + r2)

        return self.convOut(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# noise schedule (cosine, Nichol & Dhariwal) — identical formula to model.js
# ---------------------------------------------------------------------------
def make_schedule(T, device="cpu", s=0.008):
    def f(t):
        return math.cos(((t / T + s) / (1 + s)) * math.pi / 2) ** 2

    f0 = f(0)
    betas, alphas, alpha_bars = [], [], []
    prev_alpha_bar = 1.0
    for i in range(T):
        abar = f(i + 1) / f0
        beta = min(max(1 - abar / prev_alpha_bar, 1e-5), 0.999)
        betas.append(beta)
        alphas.append(1 - beta)
        alpha_bars.append(abar)
        prev_alpha_bar = abar
    return {
        "T": T,
        "betas": torch.tensor(betas, dtype=torch.float32, device=device),
        "alphas": torch.tensor(alphas, dtype=torch.float32, device=device),
        "alphaBars": torch.tensor(alpha_bars, dtype=torch.float32, device=device),
    }


def train_step(model, optimizer, batch_x0, schedule, grad_clip=1.0):
    """One optimizer step. Self-conditioning half the time, exactly as in
    model.js's trainStep (first pass under no_grad, second pass trained)."""
    B = batch_x0.shape[0]
    T = schedule["T"]
    device = batch_x0.device
    t_idx = torch.randint(0, T, (B,), device=device)
    t_norm = t_idx.float() / max(T - 1, 1)
    alpha_bar = schedule["alphaBars"][t_idx].view(B, 1, 1, 1)

    noise = torch.randn_like(batch_x0)
    sqrt_ab = alpha_bar.sqrt()
    sqrt_omab = (1 - alpha_bar).clamp_min(1e-8).sqrt()
    xt = batch_x0 * sqrt_ab + noise * sqrt_omab

    if torch.rand(()).item() < SELF_COND_PROB:
        with torch.no_grad():
            pred_noise0 = model(xt, torch.zeros_like(batch_x0), t_norm)
            self_cond = ((xt - pred_noise0 * sqrt_omab) / sqrt_ab).clamp(-X0_CLIP, X0_CLIP)
    else:
        self_cond = torch.zeros_like(batch_x0)

    pred_noise = model(xt, self_cond, t_norm)
    loss = F.mse_loss(pred_noise, noise)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = None
    if grad_clip is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
    optimizer.step()
    return loss.item(), grad_norm


@torch.no_grad()
def sample_batch(model, schedule, batch_size, device):
    """Batched reverse diffusion: `batch_size` independent samples, sharing
    the T model calls (one call scores the whole batch at once). This is the
    throughput lever — the browser's sampleOne() does one grid per T calls."""
    H, Wd, V = model.height, model.width, model.vocab_size
    T = schedule["T"]
    x = torch.randn(batch_size, H, Wd, V, device=device)
    self_cond = torch.zeros_like(x)
    for i in range(T - 1, -1, -1):
        t_norm = torch.full((batch_size,), i / max(T - 1, 1), device=device)
        alpha = schedule["alphas"][i]
        alpha_bar = schedule["alphaBars"][i]
        beta = schedule["betas"][i]

        pred_noise = model(x, self_cond, t_norm)
        sqrt_ab = alpha_bar.sqrt()
        sqrt_omab = (1 - alpha_bar).clamp_min(1e-8).sqrt()
        x0hat = ((x - pred_noise * sqrt_omab) / sqrt_ab).clamp(-X0_CLIP, X0_CLIP)

        coef1 = 1.0 / alpha.sqrt()
        coef2 = beta / sqrt_omab
        mean = (x - pred_noise * coef2) * coef1
        if i > 0:
            mean = mean + torch.randn_like(x) * beta.sqrt()
        x = mean
        self_cond = x0hat
    return x  # [B,H,W,V] continuous — decode via argmax (see data.py)


class EMA:
    """Shadow-weight EMA used only for sampling (quality-gate evals + the
    search itself) — the raw weights are what actually get trained. Standard
    diffusion-model trick; this is the single biggest lever for 'the samples
    look clean' short of more data."""

    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}
