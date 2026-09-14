#!/usr/bin/env python3
"""
Export a browser-loadable, chunked weights/ folder from any checkpoint.pt,
at any time - including while train.py is still running against the same
checkpoint (it's only ever read here, never written). Useful because
train.py's Ctrl+C path saves checkpoint.pt but currently does NOT also
export weights - this covers that gap, and also lets you peek at progress
mid-run without stopping training at all.

Output matches what train.py itself produces (see
ascii_diffusion/bridge.py's export_model_chunked and README_LOCAL.md): a
manifest.json plus several part-NNN.txt files, none of which should ever
approach GitHub's 100MB per-file push limit regardless of model size.

Usage:
    python export_checkpoint.py runs/big/checkpoint.pt
    python export_checkpoint.py runs/big/checkpoint.pt ../data/weights
    python export_checkpoint.py runs/big/checkpoint.pt ../data/weights --chunk-size-mb 10
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ascii_diffusion import model as M, bridge


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("out_dir", nargs="?", default=None,
                    help="default: a 'weights_manual' folder next to the checkpoint")
    p.add_argument("--chunk-size-mb", type=float, default=20)
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint) or ".", "weights_manual")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = M.DiffusionModel(**ckpt["cfg"])
    model.load_state_dict(ckpt["ema"])  # export EMA weights, same as train.py's own auto-export does
    model.eval()

    manifest_path, chunk_files = bridge.export_model_chunked(
        model, out_dir, chunk_size_bytes=int(args.chunk_size_mb * 1e6),
        extra_meta={"timesteps": ckpt.get("timesteps", 50), "step": ckpt["step"]},
    )
    print(f"step {ckpt['step']:,} -> wrote {len(chunk_files)} chunk(s) + manifest to {out_dir}/ "
          f"({model.count_params():,} params)")


if __name__ == "__main__":
    main()
