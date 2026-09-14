#!/usr/bin/env python3
"""
Local/GPU training for the ascii-diffusion model. Same rules as js/app.js:

  - The quality gate (are we done training?) is scored ONLY against
    dataset.train_bin. It NEVER reads dataset.test_bin. That boundary is
    the entire point of this project (see ../README.md) and nothing in
    this file crosses it.
  - Training stops when the quality gate passes, not on a fixed step count.
    --max-steps is a safety ceiling, same meaning as the browser's "Max
    steps" field.

Usage:
    python train.py --data ../data/dataset.json
    python train.py --data ../data/dataset.json --resume runs/latest/checkpoint.pt

Run `python train.py --help` for the full knob list. Defaults are picked for
"much bigger than the in-browser default (hidden=32, 6 blocks, ~50K params)"
while still training in minutes-to-an-hour on an Apple Silicon Mac via MPS —
see ../README_LOCAL.md for what to expect and how to go bigger still.
"""
import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ascii_diffusion import data, model as M, similarity, bridge


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_lambda_factory(warmup_steps, max_steps, final_ratio=0.1):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        if max_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return final_ratio + (1 - final_ratio) * cosine
    return fn


@torch.no_grad()
def evaluate_quality(eval_model, schedule, ds, n_samples, device, win=3):
    """Quality gate: score fresh samples against the TRAIN set only. Never
    touches ds.test_bin — see the module docstring."""
    eval_model.eval()
    samples = M.sample_batch(eval_model, schedule, n_samples, device)
    grids = []
    for i in range(n_samples):
        text = data.analog_to_text(samples[i], ds.vocab, ds.height, ds.width)
        grids.append(data.text_to_binary(text, ds.vocab, ds.height, ds.width))
    gen_bin = torch.tensor(grids, dtype=torch.float32, device=device)
    train_bin = torch.tensor(ds.train_bin, dtype=torch.float32, device=device)
    best_ssim, _, _ = similarity.best_match(gen_bin, train_bin, win=win)
    eval_model.train()
    return best_ssim.mean().item(), best_ssim.max().item()


def save_checkpoint(path, model, optimizer, ema, step, cfg, timesteps):
    torch.save({
        "step": step, "cfg": cfg, "timesteps": timesteps,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "ema": ema.state_dict(),
    }, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="../data/dataset.json")
    p.add_argument("--out", default="runs/default")
    p.add_argument("--chunk-size-mb", type=float, default=20,
                    help="split exported weights into files at most this big (see README_LOCAL.md - "
                         "keeps a single committed file from ever approaching GitHub's 100MB push limit, "
                         "no matter how much bigger the model gets)")
    p.add_argument("--resume", default=None, help="path to a checkpoint.pt to resume from")

    g = p.add_argument_group("model size (\"much bigger\")")
    g.add_argument("--hidden", type=int, default=128, help="channel width at every stage (browser default: 32)")
    g.add_argument("--blocks", type=int, default=8, help="bottleneck residual blocks (browser default: 6)")
    g.add_argument("--kernel", type=int, default=7, help="bottleneck conv kernel size")
    g.add_argument("--time-dim", type=int, default=64)
    g.add_argument("--timesteps", "-T", type=int, default=50, help="diffusion steps")

    t = p.add_argument_group("training")
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--warmup-steps", type=int, default=500)
    t.add_argument("--max-steps", type=int, default=100_000, help="safety ceiling, not a target (see docstring)")
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--ema-decay", type=float, default=0.9995)
    t.add_argument("--seed", type=int, default=0)

    q = p.add_argument_group("quality gate (train-only, see docstring)")
    q.add_argument("--target-quality", type=float, default=0.92)
    q.add_argument("--eval-every", type=int, default=250)
    q.add_argument("--eval-samples", type=int, default=16)

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--checkpoint-every", type=int, default=500)
    args = p.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print(f"device: {device}")
    ds = data.Dataset(args.data)
    print(ds)
    if ds.width != 20 or ds.height != 10:
        print(f"note: dataset grid is {ds.width}x{ds.height}, not the usual 20x10 — that's fine, "
              f"the model adapts, just double check that's what you intended.")

    cfg = dict(height=ds.height, width=ds.width, vocab_size=len(ds.vocab),
               hidden=args.hidden, time_dim=args.time_dim,
               bottleneck_blocks=args.blocks, bottleneck_kernel=args.kernel)
    model = M.DiffusionModel(**cfg).to(device)
    eval_model = M.DiffusionModel(**cfg).to(device)  # separate instance so eval never touches training state
    print(f"model: {model.count_params():,} params "
          f"(hidden={args.hidden}, blocks={args.blocks}, kernel={args.kernel}, T={args.timesteps})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(args.warmup_steps, args.max_steps))
    ema = M.EMA(model, decay=args.ema_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt["cfg"] != cfg:
            print(f"WARNING: checkpoint cfg {ckpt['cfg']} != current cfg {cfg} — "
                  f"resuming would load mismatched shapes and fail below.")
        if ckpt.get("timesteps", args.timesteps) != args.timesteps:
            print(f"WARNING: checkpoint was trained with T={ckpt.get('timesteps')}, "
                  f"you passed --timesteps {args.timesteps}. Using {ckpt.get('timesteps')} "
                  f"to stay consistent with the noise schedule already trained on.")
            args.timesteps = ckpt.get("timesteps", args.timesteps)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        ema.load_state_dict({k: v.to(device) for k, v in ckpt["ema"].items()})
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    # Built AFTER the resume block on purpose: resuming can override
    # args.timesteps from the checkpoint above, and the schedule must match
    # whatever T the model was actually trained with.
    schedule = M.make_schedule(args.timesteps, device=device)

    train_tensor = ds.train_tensor(device)
    n_train = train_tensor.shape[0]
    batch_size = min(args.batch_size, n_train)

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    weights_dir = os.path.join(args.out, "weights")

    step = start_step
    loss_ema = None
    t0 = time.time()
    reached_target = False

    try:
        while step < args.max_steps:
            idx = torch.randint(0, n_train, (batch_size,), device=device)
            batch = train_tensor[idx]
            loss, grad_norm = M.train_step(model, optimizer, batch, schedule, grad_clip=args.grad_clip)
            scheduler.step()
            ema.update(model)
            step += 1
            loss_ema = loss if loss_ema is None else 0.98 * loss_ema + 0.02 * loss

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                sps = (step - start_step) / max(elapsed, 1e-6)
                lr_now = scheduler.get_last_lr()[0]
                print(f"step {step:>7,} | loss {loss:.4f} (ema {loss_ema:.4f}) | "
                      f"grad_norm {grad_norm:.3f} | lr {lr_now:.2e} | {sps:.1f} steps/s", flush=True)

            if step % args.eval_every == 0:
                eval_model.load_state_dict(ema.state_dict())
                mean_q, max_q = evaluate_quality(eval_model, schedule, ds, args.eval_samples, device)
                print(f"  [quality gate] step {step:,} | mean best-SSIM vs TRAIN {mean_q*100:.1f}% "
                      f"(target {args.target_quality*100:.0f}%) | best of batch {max_q*100:.1f}%", flush=True)
                if mean_q >= args.target_quality:
                    reached_target = True
                    save_checkpoint(ckpt_path, model, optimizer, ema, step, cfg, args.timesteps)
                    bridge.export_model_chunked(eval_model, weights_dir, chunk_size_bytes=int(args.chunk_size_mb * 1e6),
                                                 extra_meta={"timesteps": args.timesteps, "step": step, "meanQuality": mean_q})
                    print(f"quality target reached at step {step:,}. "
                          f"checkpoint: {ckpt_path}  browser weights: {weights_dir}/ "
                          f"(commit this whole folder - see README_LOCAL.md)")
                    break

            if step % args.checkpoint_every == 0:
                save_checkpoint(ckpt_path, model, optimizer, ema, step, cfg, args.timesteps)

    except KeyboardInterrupt:
        print("\ninterrupted — saving checkpoint before exit")
        save_checkpoint(ckpt_path, model, optimizer, ema, step, cfg, args.timesteps)
        return

    if not reached_target:
        save_checkpoint(ckpt_path, model, optimizer, ema, step, cfg, args.timesteps)
        eval_model.load_state_dict(ema.state_dict())
        bridge.export_model_chunked(eval_model, weights_dir, chunk_size_bytes=int(args.chunk_size_mb * 1e6),
                                     extra_meta={"timesteps": args.timesteps, "step": step})
        print(f"max steps ({args.max_steps:,}) reached without hitting the quality target. "
              f"Resume with:\n  python train.py --data {args.data} --out {args.out} --resume {ckpt_path} "
              f"--max-steps {args.max_steps * 2}")


if __name__ == "__main__":
    main()
