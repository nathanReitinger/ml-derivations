"""
Batched, vectorized reimplementation of js/similarity.js's ssim()/hammingSimilarity().

The formulas are identical (box-filtered win x win windows, valid/no padding,
clamped to [0,1]) — see verify_pairwise_ssim.py for a numeric cross-check
against a line-for-line port of the original nested loops. The only thing
that's different is that this scores an entire [batch x held-out-set] matrix
in one shot instead of one (generated, held-out) pair at a time, which is
what makes a real hashcat-style search across thousands of held-out targets
and large sample batches feasible.
"""
import torch
import torch.nn.functional as F

C1 = 0.01 ** 2
C2 = 0.03 ** 2


def pairwise_ssim(gen, test, win=3, window_chunk=16):
    """
    gen:  [B, H, W] float tensor, values in {0,1}
    test: [N, H, W] float tensor, values in {0,1}
    returns: [B, N] mean-SSIM over all valid win x win windows, clamped [0,1].

    Accumulates the per-window average in chunks over the window-position
    axis instead of materializing a [num_windows, B, N] tensor (and several
    same-shaped intermediates) all at once. num_windows is small (144 for a
    20x10 grid at win=3) but num_windows * B * N does not stay small: at a
    512-sample wave against an 18,000-file held-out set that's ~1.3 billion
    elements per intermediate, several of which are alive at once — that's
    the actual cause of an MPS/CUDA out-of-memory error here, not dataset or
    model size. This produces the exact same numbers (mean of X is the same
    whether you sum-then-divide all at once or accumulate the sum across
    chunks and divide once at the end), just bounded to
    window_chunk * B * N in memory at any moment.
    """
    n = win * win
    denom_n = max(n - 1, 1)

    g = gen.unsqueeze(1)   # [B,1,H,W]
    t = test.unsqueeze(1)  # [N,1,H,W]

    g_patches = F.unfold(g, kernel_size=win)  # [B, n, num_windows]
    t_patches = F.unfold(t, kernel_size=win)  # [N, n, num_windows]
    num_windows = g_patches.shape[-1]

    sumA, sumA2 = g_patches.sum(1), (g_patches ** 2).sum(1)  # [B, num_windows]
    sumB, sumB2 = t_patches.sum(1), (t_patches ** 2).sum(1)  # [N, num_windows]
    muA, muB = sumA / n, sumB / n
    varA = (sumA2 - n * muA ** 2) / denom_n
    varB = (sumB2 - n * muB ** 2) / denom_n

    B, N = gen.shape[0], test.shape[0]
    total = torch.zeros(B, N, device=gen.device, dtype=g_patches.dtype)

    for start in range(0, num_windows, window_chunk):
        end = min(start + window_chunk, num_windows)
        gp = g_patches[:, :, start:end].permute(2, 0, 1)  # [chunk, B, n]
        tp = t_patches[:, :, start:end].permute(2, 0, 1)  # [chunk, N, n]
        sumAB = torch.bmm(gp, tp.transpose(1, 2))          # [chunk, B, N]

        muA_w = muA[:, start:end].t().unsqueeze(-1)        # [chunk, B, 1]
        muB_w = muB[:, start:end].t().unsqueeze(1)         # [chunk, 1, N]
        varA_w = varA[:, start:end].t().unsqueeze(-1)
        varB_w = varB[:, start:end].t().unsqueeze(1)

        covAB = (sumAB - n * muA_w * muB_w) / denom_n
        num = (2 * muA_w * muB_w + C1) * (2 * covAB + C2)
        den = (muA_w ** 2 + muB_w ** 2 + C1) * (varA_w + varB_w + C2)
        total += (num / den).sum(dim=0)  # fold this chunk's windows into the running total

    return (total / num_windows).clamp(0, 1)


def pairwise_hamming(gen, test):
    """gen: [B,H,W], test: [N,H,W], both in {0,1}. Returns [B,N] fraction-identical."""
    B, H, W = gen.shape
    N = test.shape[0]
    hw = H * W
    g = gen.reshape(B, hw)
    t = test.reshape(N, hw)
    same = g @ t.t() + (1 - g) @ (1 - t).t()
    return same / hw


def best_match(gen, test, win=3, window_chunk=16):
    """gen: [B,H,W]; test: [N,H,W]. Returns per-sample best SSIM/Hamming and
    the index into `test` it matched — i.e. AsciiSimilarity.bestMatch,
    vectorized over the whole batch at once."""
    ssim_mat = pairwise_ssim(gen, test, win=win, window_chunk=window_chunk)  # [B,N]
    best_ssim, best_idx = ssim_mat.max(dim=1)
    ham_mat = pairwise_hamming(gen, test)              # [B,N]
    best_ham = ham_mat.gather(1, best_idx.view(-1, 1)).squeeze(1)
    return best_ssim, best_ham, best_idx
