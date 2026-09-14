#!/usr/bin/env python3
"""
The hashcat-style search: repeatedly sample from a trained model and score
every attempt against the HELD-OUT test set only (never train — same
invariant as js/app.js's generationLoop). Stops a run the instant one
sample's best SSIM against any held-out file clears --threshold.

Unlike the browser (one grid at a time, one <canvas>+one animation-frame
yield per attempt), this samples a whole batch per "wave" and scores the
entire [batch x held-out set] SSIM matrix in one shot — see
ascii_diffusion/similarity.py. That's what makes running the search --runs
times, to report a distribution of attempts-to-match instead of a single
lucky number, actually feasible.

Usage:
    python search.py --checkpoint runs/default/checkpoint.pt --data ../data/dataset.json
    python search.py --checkpoint runs/default/checkpoint.pt --data ../data/dataset.json \\
        --runs 50 --wave 512 --threshold 0.97
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import time

import numpy as np

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ascii_diffusion import data, model as M, similarity


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_match_png(text, vocab, height, width, path, cell=16, ink=(108, 99, 255), paper=(255, 255, 255)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (width * cell, height * cell), paper)
    draw = ImageDraw.Draw(img)
    rows = text.split("\n")
    bg = vocab[0]
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch != bg:
                draw.rectangle([c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1], fill=ink)
    img.save(path)


def save_comparison_png(gen_text, heldout_text, vocab, height, width, path, caption, cell=16):
    """Generated | held-out side by side with a caption - closer to what the
    browser's own match panel shows than two separate files, and what
    actually gets displayed in the terminal (see show_image below)."""
    from PIL import Image, ImageDraw

    gap, pad, caption_h = 24, 16, 28
    tile_w, tile_h = width * cell, height * cell
    total_w = pad * 2 + tile_w * 2 + gap
    total_h = pad * 2 + tile_h + caption_h
    img = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    def draw_tile(text, x0, y0, ink):
        rows = text.split("\n")
        bg = vocab[0]
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch != bg:
                    draw.rectangle([x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell - 1, y0 + (r + 1) * cell - 1],
                                   fill=ink)
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline=(225, 225, 230))

    draw_tile(gen_text, pad, pad, (108, 99, 255))          # violet: generated
    draw_tile(heldout_text, pad + tile_w + gap, pad, (20, 158, 143))  # teal: held-out
    draw.text((pad, pad + tile_h + 6), "generated", fill=(90, 90, 100))
    draw.text((pad + tile_w + gap, pad + tile_h + 6), "held-out (never trained on)", fill=(90, 90, 100))
    draw.text((pad, total_h - 20), caption, fill=(20, 20, 25))
    img.save(path)


def show_image(path):
    """Best-effort: render inline if the terminal supports it (iTerm2's
    proprietary escape sequence), otherwise open with the OS's default
    viewer. Never raises - a display failure shouldn't break the search."""
    import base64
    import platform
    import subprocess

    try:
        if os.environ.get("TERM_PROGRAM") == "iTerm.app":
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            sys.stdout.write(f"\033]1337;File=inline=1;preserveAspectRatio=1:{data}\a\n")
            sys.stdout.flush()
            return True
        opener = {"Darwin": "open", "Linux": "xdg-open"}.get(platform.system())
        if opener:
            subprocess.run([opener, path], check=False,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except Exception:
        pass
    return False


@torch.no_grad()
def run_search(model, schedule, test_bin, vocab, threshold, wave, max_attempts, device, win=3,
                window_chunk=16, progress_every=1):
    """One independent search run. Returns a dict with attempts/elapsed/match info,
    or attempts=None if max_attempts was hit with no match. Prints progress every
    `progress_every` waves - a single run can be hundreds of waves, and with no
    output at all in between that's indistinguishable from a hang."""
    attempts = 0
    t0 = time.time()
    best_ever = -1.0
    wave_i = 0
    while attempts < max_attempts:
        samples = M.sample_batch(model, schedule, wave, device)  # [wave,H,W,V]
        texts = [data.analog_to_text(samples[i], vocab, model.height, model.width) for i in range(wave)]
        # np.array(...) first, then one tensor conversion - converting a raw list of
        # ndarrays directly is a known-slow path torch warns about.
        grids = torch.tensor(
            np.array([data.text_to_binary(t, vocab, model.height, model.width) for t in texts]),
            device=device, dtype=torch.float32,
        )
        best_ssim, best_ham, best_idx = similarity.best_match(grids, test_bin, win=win, window_chunk=window_chunk)
        best_ssim_cpu = best_ssim.cpu()
        local_best = best_ssim_cpu.max().item()
        if local_best > best_ever:
            best_ever = local_best
        hits = (best_ssim_cpu >= threshold).nonzero(as_tuple=True)[0]
        wave_i += 1
        if len(hits) > 0:
            j = hits[0].item()
            return {
                "attempts": attempts + j + 1,
                "elapsed": time.time() - t0,
                "ssim": best_ssim_cpu[j].item(),
                "hamming": best_ham[j].item(),
                "test_index": best_idx[j].item(),
                "generated_text": texts[j],
                "best_ever": best_ever,
            }
        attempts += wave
        if progress_every and wave_i % progress_every == 0:
            elapsed = time.time() - t0
            rate = attempts / max(elapsed, 1e-6)
            print(f"    ...{attempts:,}/{max_attempts:,} attempts, {elapsed:.0f}s elapsed "
                  f"({rate:.0f}/s) | best so far this run: {best_ever*100:.1f}% (need {threshold*100:.0f}%)",
                  flush=True)
    return {"attempts": None, "elapsed": time.time() - t0, "best_ever": best_ever}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="../data/dataset.json")
    p.add_argument("--out", default="runs/default/search")
    p.add_argument("--threshold", type=float, default=0.97)
    p.add_argument("--wave", type=int, default=512, help="samples generated+scored per batch")
    p.add_argument("--window-chunk", type=int, default=16,
                    help="SSIM windows processed at once inside the scorer - lower this if you hit an "
                         "out-of-memory error (MPS/CUDA); it trades a bit of speed for a lot less peak memory")
    p.add_argument("--max-attempts", type=int, default=200_000, help="per-run safety cap")
    p.add_argument("--progress-every", type=int, default=1,
                    help="print a progress line every N waves (0 to disable)")
    p.add_argument("--runs", type=int, default=20, help="independent runs, to report a distribution")
    p.add_argument("--timesteps", "-T", type=int, default=None, help="override checkpoint's T if set")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--save-examples", type=int, default=3, help="how many matched pairs to save as PNGs")
    p.add_argument("--show", action="store_true", default=True, help="display each saved match image (default on)")
    p.add_argument("--no-show", dest="show", action="store_false", help="save PNGs without trying to display them")
    p.add_argument("--showcase-out", default=None,
                    help="also write a compact result here for the browser to show (e.g. ../data/showcase.json). "
                         "Uses the best (lowest-attempts) match found, if any.")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["cfg"]
    model = M.DiffusionModel(**cfg).to(device)
    model.load_state_dict(ckpt["ema"])  # sample with EMA weights, same as the quality gate does
    model.eval()

    ds = data.Dataset(args.data)
    T = args.timesteps or ckpt.get("timesteps", 50)
    schedule = M.make_schedule(T, device=device)
    test_bin = torch.tensor(ds.test_bin, dtype=torch.float32, device=device)

    print(f"device: {device} | model step {ckpt['step']:,} | {model.count_params():,} params | T={T}")
    print(f"held-out test set: {test_bin.shape[0]} files | threshold {args.threshold} | wave {args.wave} | runs {args.runs}")

    results = []
    for run_i in range(args.runs):
        r = run_search(model, schedule, test_bin, ds.vocab, args.threshold, args.wave, args.max_attempts, device,
                        window_chunk=args.window_chunk, progress_every=args.progress_every)
        results.append(r)
        if r["attempts"] is not None:
            print(f"run {run_i+1:>3}/{args.runs}: matched after {r['attempts']:>7,} attempts "
                  f"({r['elapsed']:.1f}s) | {r['ssim']*100:.1f}% SSIM vs test[{r['test_index']}] "
                  f"| {r['hamming']*100:.1f}% identical", flush=True)
        else:
            print(f"run {run_i+1:>3}/{args.runs}: NO MATCH within {args.max_attempts:,} attempts "
                  f"(best seen: {r['best_ever']*100:.1f}% SSIM)", flush=True)

    matched = [r["attempts"] for r in results if r["attempts"] is not None]
    summary = {
        "runs": args.runs, "threshold": args.threshold, "wave": args.wave,
        "max_attempts": args.max_attempts, "matched_runs": len(matched),
        "unmatched_runs": args.runs - len(matched),
    }
    if matched:
        summary.update({
            "attempts_mean": statistics.mean(matched),
            "attempts_median": statistics.median(matched),
            "attempts_min": min(matched),
            "attempts_max": max(matched),
            "attempts_stdev": statistics.stdev(matched) if len(matched) > 1 else 0.0,
        })
    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "runs": results}, f, indent=2)

    saved = 0
    best_match_record = None  # lowest-attempts match across all runs, for --showcase-out
    for r in results:
        if r["attempts"] is None or saved >= args.save_examples:
            continue
        gen_text = r["generated_text"]
        test_text = ds.test_texts[r["test_index"]]
        base = os.path.join(args.out, f"match_{saved}")
        with open(base + "_generated.txt", "w") as f:
            f.write(gen_text + "\n")
        with open(base + "_heldout.txt", "w") as f:
            f.write(test_text + "\n")
        caption = (f"{r['ssim']*100:.1f}% SSIM · {r['hamming']*100:.1f}% identical · "
                   f"{r['attempts']:,} attempts · test[{r['test_index']}]")
        comparison_path = base + "_comparison.png"
        try:
            save_match_png(gen_text, ds.vocab, ds.height, ds.width, base + "_generated.png")
            save_match_png(test_text, ds.vocab, ds.height, ds.width, base + "_heldout.png")
            save_comparison_png(gen_text, test_text, ds.vocab, ds.height, ds.width, comparison_path, caption)
            if args.show:
                show_image(comparison_path)
        except Exception as e:
            print(f"(skipped PNG render: {e})")
        if best_match_record is None or r["attempts"] < best_match_record["attempts"]:
            best_match_record = {**r, "test_file": ds.test_files[r["test_index"]]}
        saved += 1
    print(f"\nwrote summary + {saved} example match(es) to {args.out}/")

    if args.showcase_out and best_match_record:
        r = best_match_record
        showcase = {
            "generatedText": r["generated_text"],
            "heldOutText": ds.test_texts[r["test_index"]],
            "heldOutFile": r["test_file"],
            "ssim": r["ssim"],
            "hamming": r["hamming"],
            "attempts": r["attempts"],
            "elapsedSeconds": r["elapsed"],
            "threshold": args.threshold,
            "modelStep": ckpt["step"],
            "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(args.showcase_out) or ".", exist_ok=True)
        with open(args.showcase_out, "w") as f:
            json.dump(showcase, f, indent=2)
        print(f"wrote showcase result to {args.showcase_out} (best of this run: {r['attempts']:,} attempts, "
              f"{r['ssim']*100:.1f}% SSIM)")
    elif args.showcase_out:
        print(f"--showcase-out given but no run in this batch found a match - nothing written to {args.showcase_out}")
    print(f"\nwrote summary + {saved} example match(es) to {args.out}/")


if __name__ == "__main__":
    main()
