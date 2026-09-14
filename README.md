# ascii diffusion

A small, deliberately well-trained diffusion model that trains **in the
visitor's browser** (TensorFlow.js, no server, no backend) on a subset of
tiny 20×10, 2-symbol ascii drawings — however many `data/dataset.json` says
are `train` (2,000 by default with the bundled generator; the browser just
reads whatever's there). The rest of the pack — every other file — is held
out and never shown to the model during training. Training doesn't stop on
a step count; it stops once a quality gate confirms the model is actually
good, checked strictly against the training set. Only then does generation
start: the model samples new drawings one at a time, each one scored by
SSIM against the held-out set, and generation stops the instant one lands
close enough to a drawing it never trained on.

**That's the thing this site demonstrates**: shrink the problem space far
enough — a small grid, two symbols, repetitive parametric shapes — and a
generative model's own outputs can land close to data it was never trained
on. Not because it memorized that specific file, but because there are only
so many ways to draw, say, a ring of a given radius on a 20×10 grid, and a
model that's actually learned that *family* well will eventually produce an
instance close enough to one it never saw.

Ships with a small generated placeholder pack (66 simple parametric shapes —
boxes, rings, stripes, crosses...) so the site works immediately with no
setup. For anything beyond a quick look, run `python g.py` once to replace
it with a much bigger, denser pack (20,000 files, 2,000 train / 18,000 held
out by default) — see "Using your own data" below. Either script leaves you
with a fresh `data/dataset.json`; the browser just reads whatever's there.

## What changed in this version

The previous version of this site did the same kind of thing at a much
larger scale (60×30 grid, a ~2,600-file pack of flags/emoji/fractals/words,
exact character-for-character matching) and trained for a fixed step count.
This version is smaller and stricter in a few specific ways:

- **Pack**: `ascii_art_pack___small`, 20×10 characters, 2 symbols (was
  `ascii_art_pack___large`, 60×30).
- **Split**: `scripts/build_dataset.js` now takes a fixed-size **train**
  subset (2,000 files by default, configurable — was 50 in the very first
  cut of this version, before the pack itself grew) instead of a 50/50
  split. Everything else becomes the held-out **test** set — same role as
  before, just no longer exactly half the pack.
- **Training stops on a quality gate, not a step count.** Every
  `hp-eval-every` steps, the half-trained model generates a few samples and
  each is scored (SSIM) against its nearest **training** image only.
  Training keeps going until the mean of those scores clears a target
  (default 90%), or the step ceiling is hit — whichever comes first. This is
  what "actually good" means here, made concrete instead of left to an
  arbitrary step count picked in advance. If the ceiling is hit first, "Train
  more" resumes the same model rather than starting over.
- **Generation matches on closeness, not equality.** Each generated sample
  is scored against every held-out test image with SSIM (`js/similarity.js`,
  new in this version) plus a plain character-identity percentage.
  Generation stops the moment one sample's best score clears the match
  threshold (default 97%, adjustable live). The live "best so far" readout
  shows the closest attempt at any given moment, hashcat-style.

**The one rule that makes any of this mean something**: the quality gate
during training reads only `state.trainSet`, and the generation loop's
matching reads only `state.testSet` (see the comments in `js/app.js` and
`scripts/build_dataset.js`). If the quality gate were allowed to peek at the
held-out set to decide when training is "good enough," that would leak test
information into training and the final "it never saw this" claim would be
false. The two arrays are built once in `init()` and never cross into each
other's function.

Everything below that isn't specific to those changes — the model
architecture, the visual theme, the build-id tracking — is unchanged from
the previous version and described here for completeness.

## Local / GPU training (new)

Training in a browser tab is a real constraint: no checkpointing, no
running overnight, a single tab babysitting one model, and no real GPU path
on Node (no CUDA on Mac; tf.js's WebGL backend is what the browser itself
already uses, which is fine for the bundled placeholder pack but not for
"much bigger"). `python/` is a from-scratch PyTorch port of the exact same
model, kept in exact numeric sync with `js/model.js` — same architecture,
same noise schedule, same LayerNorm epsilon, same padding, formula for
formula — plus a few standard upgrades that only make sense with real
compute (EMA weights, gradient clipping, LR warmup) and a weight bridge that
lets a model trained there load straight into this browser page.

- **Train** a much bigger model with GPU acceleration (MPS on Apple
  Silicon, CUDA on Linux/Windows, CPU otherwise, auto-detected):
  `python/train.py`.
- **Search**: run the same hashcat-style SSIM search `js/app.js` does,
  vectorized and batched, many independent times to get a real distribution
  of attempts-to-match instead of one anecdote: `python/search.py`.
- **Load the result back into this page**: `python/train.py` exports a
  `weights/` folder (a small manifest plus several `part-NNN.txt` chunks —
  see [README_LOCAL.md](README_LOCAL.md) for why it's split) alongside its
  checkpoint; commit that folder as `data/weights/` and the page loads it
  **automatically** — visitors land on "Use existing model" already active,
  no click needed. A "Train your own" button next to it switches to the
  original full in-browser training UI at any time; a file picker under
  "Use existing model" also accepts loading a model directly from disk
  without committing it: either a single legacy `weights.json`, or
  `manifest.json` selected together with all of its `part-NNN.txt` files
  (select the whole `weights/` folder's contents at once, e.g. Cmd/Ctrl-A
  inside it) — picking `manifest.json` by itself isn't enough on its own,
  since the weights themselves live in the part files.

Full walkthrough, flags, and what to actually expect timing-wise:
**[README_LOCAL.md](README_LOCAL.md)**.

## Model

The model in `js/model.js` is a small conv-net predicting noise at each
diffusion step (DDPM, cosine schedule):

- **Self-conditioning** (Chen, Zhang & Hinton, *Analog Bits*) — the model
  sees its own previous estimate of the clean image half the time during
  training, which is what makes the "analog bits" (signed one-hot) encoding
  in `js/data.js` produce sharp, coherent samples.
- **FiLM conditioning at every stage** — the timestep embedding modulates
  (scale + shift) the stem, the downsampled path, every bottleneck block,
  and the refine blocks.
- **Per-pixel layer normalization** before each FiLM step.
- **A 6-block residual bottleneck** with kernel-7 convolutions, giving the
  bottleneck a receptive field that covers most of the 20×10 grid even
  though it operates at half resolution.
- **A cosine noise schedule**, which spends the available diffusion steps
  (T, default 40) more usefully than a linear schedule at low step counts.

`buildModel`, `trainStep`, and `sampleOne` are unchanged in shape from the
large-pack version — this rework only touches what happens *around* those
calls (`js/app.js`), plus the new `js/similarity.js`.

### What to actually expect

At 20×10 with 2 symbols and 50 training images, this is a genuinely small,
learnable space — the model isn't fighting the diversity a 1,300-image pack
of flags/emoji/fractals/words presented. Repetitive, parametric shapes
(boxes, rings, stripes, crosses) both clear the training quality gate
fastest *and* produce close matches against held-out examples soonest,
because a well-learned parametric family is exactly the situation where an
unseen instance can coincide closely with something the model generalized
to. Anything more irregular in your own pack will take longer at both
stages, and some very specific one-off drawings may never be closely
matched no matter how long you let it run — that's the size of that
particular haystack, not a bug.

## Visual theme + build tracking

Unchanged from the previous version: a light glassmorphism theme (frosted
panels, soft blurred color instead of grid lines), `Roslindale` as the
intended display face with `Fraunces` as the free fallback (see the
commented `@font-face` block near the bottom of `css/style.css` if you own a
license), and a single motion moment (a freshly generated tile popping in).
This version adds a few new pieces in the same language: the match-threshold
field, the quality-gate's compact preview strip, and an amber "new best"
tile highlight (green is reserved for an actual match).

`js/build.js` exports `window.BUILD_ID`, shown in the topbar. Bump it every
time you push, in the same commit as the change — it's how you confirm a
deploy actually landed, independent of what a browser or CDN cache thinks is
current. `/VERSION` at the repo root is the single source of truth for the
current version number (currently `2.1`) — `js/build.js` and
`package.json`'s `version` field are kept in sync with it by hand (there's
no build step to do this automatically). Bump all three together.

## How it works

- **Data**: every `.txt` file in `data/ascii_art_pack___small/` must be
  exactly 10 rows of 20 characters, using exactly 2 distinct characters.
- **Split**: `scripts/build_dataset.js` reads that folder, validates
  dimensions, builds the vocabulary, and does a **deterministic** selection
  (seeded shuffle of the sorted filename list) of a `train` subset — 2,000
  files by default (capped at half the pack if the pack itself is small, so
  a quick placeholder pack doesn't end up almost entirely "train"; see the
  comment next to `TRAIN_SIZE` in `scripts/build_dataset.js`). Everything
  else becomes `test`, held out. Re-running the script against the same
  files always reproduces the same split. Output goes to `data/dataset.json`,
  which the page fetches at load time.
- **Model**: see "Model" above.
- **Training**: runs entirely client-side. Random mini-batches are drawn
  from the training set only. Periodically it pauses to check itself —
  see "Quality gate" below.
- **Quality gate**: every `hp-eval-every` steps, the model draws
  `hp-eval-samples` fresh generations and scores each against its nearest
  **training** image (SSIM). Training stops once the mean of those scores
  clears `hp-train-target`, or the step ceiling (`hp-steps`) is hit first.
  This never looks at the test set.
- **Generation**: each full reverse-diffusion sample is decoded to a 20×10
  text grid and scored with `js/similarity.js` against every **held-out
  test** image. The score is mean SSIM over 3×3 sliding windows (Wang et
  al., 2004), clamped to [0, 1], computed directly on the two symbols as a
  binary 0/1 grid — plus a plain character-identity percentage shown
  alongside it. The moment one sample's best score clears the match
  threshold (`hp-threshold`), generation stops and the page shows the
  generated grid next to the held-out file it matched.
- You can expand "View training set" / "View held-out test set" on the page
  itself to see exactly which files ended up in which half.

## Using your own data

**Fastest path to a bigger pack**: `python g.py` (re)generates a 20,000-file
pack across the 13 built-in shape families and writes `data/dataset.json`
directly — no separate build step needed. Pass a different total/train
size as arguments: `python g.py 5000 500`.

To use your own hand-made shapes instead:

1. Delete the placeholder files in `data/ascii_art_pack___small/` and copy
   your real pack in there — every file exactly 10 rows of 20 characters,
   using exactly 2 distinct characters.
2. Commit and push. If your repo already has a GitHub Action that runs
   `node scripts/build_dataset.js` on push, nothing about it needs to
   change — it just calls the script by path, and the script's interface
   (arguments, output file) is unchanged. (That workflow file isn't part of
   this delivery since it wasn't part of what was handed over — carry your
   existing one across as-is.)
3. In your repo's **Settings → Pages**, set **Source: GitHub Actions** (only
   needs doing once).

To rebuild and preview locally instead:

```bash
node scripts/build_dataset.js   # writes data/dataset.json
python3 -m http.server 8080     # or: npx serve
# open http://localhost:8080
```

(`fetch()` needs an actual HTTP server — opening `index.html` directly as a
`file://` URL won't load `dataset.json`.)

`build_dataset.js` takes optional arguments if you want a different source
folder, output path, or training-subset size:

```bash
node scripts/build_dataset.js [sourceDir] [outFile] [trainSize]
# e.g. train on 500 files instead of the default 2,000, holding the rest out:
node scripts/build_dataset.js data/ascii_art_pack___small data/dataset.json 500
```

If you don't have a pack yet, `node scripts/generate_placeholder_pack.js
[outDir]` will (re)generate the bundled placeholder set of 66 parametric
shapes — the same one this repo ships with.

## Pushing to GitHub without hitting a size wall

The repo itself should stay small — `data/dataset.json` is a few MB and
everything else is source. The one thing that *can* blow past GitHub's
100MB hard per-file limit is a trained model: `data/weights/` (base64
float16, split into ~20MB chunks by default — see
[README_LOCAL.md](README_LOCAL.md)) is built specifically so no single
file in it ever approaches that limit, however big the model gets. The one
file that still can is `python/runs/*/checkpoint.pt` (full precision +
optimizer state, easily 200MB+), and **it isn't needed for the live site at
all** — only `data/weights/` is. `.gitignore` already excludes `*.pt` and
the regeneratable raw pack for this reason.

If you've already tried adding one of those and hit a push error
suggesting Git LFS: don't use it here — **GitHub Pages serves Git-LFS-
tracked files as a small pointer stub, not the real content** (a permanent
Pages limitation, not a bug), so LFS would fix the push and then break the
live site. Check whether a large file already made it into your local
history:

```bash
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | sort -k3 -n -r | head -10
```

If nothing over ~50MB shows up, you're clean — just make sure the big files
are gitignored (not staged) and commit/push normally. If something big
*is* in there, the commit that added it needs to stop existing, not just
be deleted in a later commit (Git still ships the earlier commit's blob on
every push). For a solo/early-stage repo like this, the simplest fix is
usually to just start the history over rather than fight it:

```bash
rm -rf .git
git init
git add .
git commit -m "ascii diffusion: local/GPU training pipeline"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main --force
```

(`--force` is safe here specifically because you're replacing the whole
history of a repo with no other collaborators pulling from it — check that
premise before reaching for `--force` on anything you don't fully own.)

## Tuning

All of these are adjustable on the page itself, and describe **in-browser**
training — `python/train.py` has its own, much bigger defaults (hidden=128,
8 bottleneck blocks vs. these 32/6) suited to a GPU instead of a browser
tab; see [README_LOCAL.md](README_LOCAL.md).

| Parameter | Default | Panel | Notes |
|---|---|---|---|
| Max steps | 6000 | Training | A safety ceiling, not a target — see "Quality gate" above. |
| Batch size | 16 | Training | Capped to your train-set size automatically. |
| Hidden units | 32 | Training | Width of the conv layers. |
| Diffusion steps (T) | 40 | Training | Reverse steps per generated sample. |
| Learning rate | 0.002 | Training | Adam. |
| Target quality (SSIM) | 0.90 | Quality gate | Mean best-SSIM against the **training** set, across a batch of fresh samples, required to stop training. |
| Check every (steps) | 200 | Quality gate | How often the model checks itself. |
| Samples per check | 6 | Quality gate | More samples = a steadier (slower) quality estimate. |
| Match threshold (SSIM) | 0.97 | Generations | How close a generation must land to a **held-out** image to count as a match. |

The page reports which TensorFlow.js backend it's using (`webgl` is what you
want; `cpu` will be dramatically slower — shown as a warning if it happens)
and the model's parameter count once training starts.

An exact or near-exact match on a held-out file is a real event, not
guaranteed on any timescale — it depends on how repetitive/parametric your
shapes are, how well the model trained, and the match threshold you set (see
"What to actually expect" above). There's a safety pause after 20,000
generation attempts with no match, after which "Start generating" resumes
it.

## Files

```
VERSION                            current version number (2.1) - see js/build.js
index.html                        page markup
css/style.css                     styling
js/build.js                       window.BUILD_ID — bump on every deploy, shown in the topbar
js/model.js                       diffusion model (build/train/sample) — tf.js; loadWeights() for pretrained checkpoints
js/data.js                        text ↔ tensor encode/decode
js/render.js                      grid → canvas pixel rendering
js/similarity.js                  SSIM + character-identity scoring
js/app.js                         UI wiring; quality gate reads train only, matching reads test only
scripts/build_dataset.js          Node: builds data/dataset.json (train / held-out test split)
scripts/generate_placeholder_pack.js  Node: (re)generates the bundled placeholder pack
g.py                               Python: (re)generates a much bigger pack + dataset.json directly (run from repo root)
data/ascii_art_pack___small/      your source .txt files (placeholder shapes included)
data/dataset.json                 generated — do not hand-edit
python/                           local/GPU training pipeline — see README_LOCAL.md
  train.py                        CLI: train a bigger model with GPU acceleration
  search.py                       CLI: batched hashcat-style SSIM search, many runs -> a distribution
  requirements.txt
  ascii_diffusion/                model.py, data.py, similarity.py, bridge.py (the weight-export/import bridge to js/model.js)
tests/                            node-side regression tests (npm test) — model port, weight bridge, full DOM flow
package.json                      dev-only deps for tests/ (the site itself needs no npm install)
README_LOCAL.md                   local/GPU pipeline walkthrough
.github/workflows/deploy.yml      CI: rebuild + deploy on push (not included here — carry over your existing one; it needs no changes)
```
