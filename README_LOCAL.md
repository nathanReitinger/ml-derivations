# Local / GPU training

This is the walkthrough for `python/` — training a much bigger version of
the model in `js/model.js` on your own machine (GPU-accelerated on Apple
Silicon via MPS, or CUDA on Linux/Windows, falling back to CPU), and,
optionally, loading the result back into the browser page. See the main
[README.md](README.md) for what the project as a whole demonstrates and the
train/held-out-test invariant that all of this still respects.

## Why this exists, in one paragraph

A browser tab is a real constraint: no checkpointing, no running overnight,
one tab babysitting one model, and (on a Mac specifically) no real GPU path
for Node — tf.js in an actual browser already uses your GPU via WebGL, but
there's no equivalent for a plain Node process, and a browser tab isn't a
great place to train something you want to leave running for an hour.
`python/` is a deliberate, formula-for-formula PyTorch port of the exact
same model (same architecture, noise schedule, LayerNorm epsilon, padding —
verified numerically against a live tf.js instance while building this, not
just "should be equivalent"), so a checkpoint trained here can be exported
and loaded straight into the browser page's own generation/matching UI.

## Setup

```bash
cd python
pip install -r requirements.txt
```

On macOS this installs a normal PyTorch build with MPS (Metal) support —
nothing special to ask for. `train.py --device auto` (the default) checks
`torch.backends.mps.is_available()` first, then CUDA, then falls back to
CPU. The first line `train.py` prints is `device: ...` — if you expected GPU
and it says `cpu`, something's wrong with the torch install, not with this
code.

If you're on Linux and reach for `pip install torch`: the default PyPI
wheel bundles the full CUDA runtime as separate `nvidia-*` packages, which
adds up to several GB even if you only ever run on CPU. That's irrelevant
on Mac (no such bundle exists there) but worth knowing if you hit a
surprising disk-space wall on a Linux box.

## 1. Get a dataset

From the repo root (not `python/`):

```bash
python g.py            # writes data/ascii_art_pack___small/ + data/dataset.json
# or, for a different size:
python g.py 5000 500   # 5,000 total files, 500 of them train
```

`python/train.py` and `python/search.py` both just read `data/dataset.json`
directly — same file the browser fetches, same train/test split, no
separate step needed on top of what you already ran for the browser demo.

## 2. Train

```bash
cd python
python train.py --data ../data/dataset.json --out runs/big
```

This is the same "quality gate, not step count" design as the browser (see
main README) — it stops when a batch of fresh samples clears
`--target-quality` mean best-SSIM against the **train** set (never test;
see the invariant note in `train.py`'s docstring), or at `--max-steps` if
that comes first, in which case it prints the exact `--resume` command to
pick back up ("Train more", the CLI version).

Defaults are deliberately much bigger than the browser's (hidden=128 vs.
32, 8 bottleneck blocks vs. 6 — roughly two orders of magnitude more
parameters) plus a few things that only pay off with real training budget:

- **EMA weights** — a slow-moving average of the weights, used only for
  sampling (quality-gate checks and the final export). This is the standard
  trick for cleaner diffusion samples; the raw weights are what actually
  get trained, EMA just smooths what you sample *from*.
- **Gradient clipping + LR warmup** — bigger/deeper nets are more prone to
  early instability than the tiny in-browser one; both are cheap insurance.

Watch the printed `steps/s` for a few hundred steps before assuming a given
config is "too slow" — first-call kernel compilation on MPS/CUDA makes the
first handful of steps unrepresentative. If it stays slow, the model is
probably too big for your GPU's memory at that batch size; `--batch-size`
and `--hidden` are the first two knobs to turn down.

Run `python train.py --help` for the full flag list (model size, LR
schedule, quality-gate target/cadence, checkpoint frequency). Two outputs
land in `--out`:

- `checkpoint.pt` — full training state (weights, optimizer, EMA, step) —
  resume with `--resume runs/big/checkpoint.pt`.
- `weights/` — a folder: `manifest.json` (small - just the model config
  and a list of chunk files) plus `part-000.txt`, `part-001.txt`, ... -
  EMA weights only, split into pieces at most `--chunk-size-mb` (default 20)
  each. Rewritten every time the quality gate passes or the step ceiling is
  hit. Split specifically so committing it to git never risks hitting
  GitHub's 100MB per-file limit no matter how big the model gets — see "A
  note on file size" below. `js/app.js` fetches and reassembles the whole
  folder automatically; nothing needs combining by hand.

## 3. Search (the hashcat part)

```bash
python search.py --checkpoint runs/big/checkpoint.pt --data ../data/dataset.json
```

Same rule as the browser's generation loop: every attempt is scored only
against the **held-out test set**, never train. The difference is
throughput — this samples a whole batch ("wave") at once and scores the
entire `[wave x held-out set]` SSIM matrix in one vectorized pass (see
`ascii_diffusion/similarity.py`), instead of one grid + one browser repaint
per attempt.

`--runs 20` (the default) repeats the whole search 20 independent times and
reports a **distribution** of attempts-to-match — median, mean, min, max —
rather than one number from one lucky (or unlucky) run. That's the honest
version of "how many generations does it take": a single run's attempt
count is noisy; twenty of them tell you something. Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--threshold` | 0.97 | SSIM a sample must clear against *some* held-out file |
| `--wave` | 512 | samples generated + scored per batch |
| `--max-attempts` | 200,000 | per-run safety cap |
| `--runs` | 20 | independent repeats, for the distribution |
| `--save-examples` | 3 | matched (generated, held-out) pairs saved as PNGs + .txt |
| `--window-chunk` | 16 | SSIM windows scored at once inside the matcher. Lower this (try 4 or 8) if you hit an MPS/CUDA out-of-memory error — it trades a little speed for a lot less peak memory, with identical results either way. The memory cost scales with `window_chunk × wave × held-out-set-size`, so a big `--wave` against a big held-out set is the combination that needs a smaller chunk. |

Output goes to `--out` (default `runs/default/search/`): `summary.json`
(the full per-run data plus the aggregate stats) and the example matches.

**On model size vs. search speed** — worth having in mind before assuming
"bigger model" will also mean "fewer attempts": it mostly won't, on its
own. A bigger model trained well mostly buys *cleaner* samples. How many
attempts it takes to land on a specific held-out file is driven much more
by (a) how densely each shape family is covered in `train` (a family the
model has seen many examples of has a tighter, better-learned distribution
to sample from) and (b) the size of the held-out pool itself (more files =
more "shots on goal" per attempt) and (c) `--threshold`. If you want the
search itself to get faster, `g.py`'s dataset size/train-size arguments are
usually the more direct lever than the model's `--hidden`/`--blocks`.

## 4. Load the result back into the browser

Copy the whole exported folder where the page can fetch it:

```bash
cp -r runs/big/weights ../data/weights
```

The page checks for `data/weights/manifest.json` automatically on load —
**no click needed**. If found, it fetches every chunk, reassembles them,
builds a browser model matching the config in `manifest.json` (hidden/
timeDim/bottleneck size — it doesn't have to match whatever's in the
hyperparameter fields, those are only for in-browser training), calls
`js/model.js`'s `loadWeights()`, and lands on "Use existing model" already
active and ready to generate. A "Train your own" button switches to the
original in-browser training flow at any time. Generation afterward is
unchanged — still scored only against `state.testSet`, still stops on the
same match threshold.

If `data/weights/` doesn't exist, the page falls back to checking for a
legacy single `data/weights.json` (from before this became chunked), then
finally to "Train your own" if neither is found — never treated as an
error, just a normal empty-repo state.

If a loaded model's config doesn't match the currently-loaded dataset's
grid size or vocab (e.g. you trained on a different `dataset.json` than
the one the page has open), the status line says so instead of silently
loading something inconsistent.

## Showing the result (terminal + online)

`search.py --show` (on by default) saves a combined generated/held-out
comparison PNG for each matched run and displays it immediately — inline in
iTerm2 if you're using it, otherwise via the OS's default image viewer.
Use `--no-show` to skip this (e.g. over SSH with no display).

To show a specific result on the **live site itself**, without a visitor
needing to train or search anything:

```bash
python search.py --checkpoint runs/big/checkpoint.pt --data ../data/dataset.json \
  --runs 5 --showcase-out ../data/showcase.json
```

The page checks for `data/showcase.json` on load and, if present, shows it
in a "A match, found locally" panel right away — clearly labeled as a
precomputed result, not something generated live in the visitor's browser.
Commit that file alongside `data/weights/` and it just works on Pages.

## A note on file size (why this doesn't need Git LFS)

Two things keep a trained model out of Git-LFS territory:

1. `bridge.py` packs weight values as base64-encoded float16, not JSON
   arrays of numbers — for this model's ~14M parameters that's roughly
   **38MB total** instead of ~185MB. Precision cost is negligible for this
   purpose (median relative error ~0.02% per weight, well under the noise
   from EMA averaging alone).
2. `train.py` then splits that ~38MB across several `part-NNN.txt` files
   (20MB each by default, via `--chunk-size-mb`), plus a small manifest. No
   *single committed file* ever approaches GitHub's 100MB hard block —
   train a model 10x bigger and you get 10x the chunks, not one file
   quietly creeping toward the wall again.

Both matter because **Git LFS is not a fix for a file that needs to be
served from Pages**: GitHub Pages serves LFS-tracked files as their small
pointer stub, not the real content (a permanent, documented limitation,
not a bug to work around). Keeping every real file small enough to commit
normally, in as many pieces as it takes, is what actually works.

`checkpoint.pt` (full precision + optimizer + EMA state, several hundred MB)
is a different story - it's not needed for the live site at all, only for
resuming training or running `search.py` locally, so it's `.gitignore`'d
rather than shrunk. See the main README's git setup instructions if you're
seeing size-related push errors.

## Different grid sizes (small vs. large)

Everything above defaults to the 20×10 grid. `g.py` also takes a third
argument - `small` (default), `large`, or any integer scale factor:

```bash
python g.py 20000 2000 small   # 20x10 - the default
python g.py 20000 2000 large   # 60x30 - 3x block-upscale of the same shapes
python g.py 20000 2000 5       # 100x50 - or any scale you want
```

Every shape is generated at the same 20×10 reference math either way (the
per-family parameter tuning is untouched) and then block-upscaled — a
"large" ring is the exact same ring, just bigger, not a re-tuned shape with
its own random ranges. `scripts/build_dataset.js` infers the grid size from
the pack itself, so it works unmodified against either size. `train.py` and
`search.py` need no changes at all — `height`/`width` come from
`dataset.json`. The only cost of going bigger: the model's stem/refine
convs and the SSIM windows in `similarity.py` both scale with grid area, so
a 60×30 run costs roughly 9x the compute of 20×10 at the same `--hidden`.

## Regression tests

`tests/` has node-side tests covering the parts that would be easy to break
silently: the PyTorch↔tf.js weight bridge (round-trips a model's weights
through the exact JSON shape `bridge.py` produces and checks the forward
pass matches exactly), the batched sampler, and a full jsdom-driven run of
the actual `index.html`/`app.js` (dataset load → train → generate → match →
load-pretrained-weights). From the repo root:

```bash
npm install   # dev-only deps for the tests themselves; the site needs none of this
npm test
```

Worth re-running after any change to `js/model.js`, `python/ascii_diffusion/model.py`,
or `python/ascii_diffusion/bridge.py` specifically — that trio has to stay
in exact numeric agreement for a checkpoint trained in Python to mean
anything once loaded in the browser.
