import math
import random
import json
import os
import shutil
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# 20x10, 2-symbol ascii pack generator + dataset.json builder.
#
# The version this replaces was written for a much bigger canvas (centers
# around cx=30/cy=15 and radii of 8-13 only make sense on something like a
# 60x30 grid) and was never rescaled when W/H were dropped to 20x10 — every
# shape's math pointed mostly off-canvas, and gen_tree/gen_house indexed
# rows up to y=25 on a 10-row grid, which crashes immediately (confirmed:
# IndexError in gen_house on the very first run). Everything below is
# rescaled to actually fit a 20x10 canvas, and every generator writes
# through a single bounds-checked set_px() so nothing can index out of
# range again regardless of what parameters you hand it.
#
# This script both generates the raw .txt pack AND writes data/dataset.json
# directly (same schema app.js expects: height, width, vocab, train, test),
# so `python g.py` alone is enough to go straight to training on the site.
# If you'd rather use scripts/build_dataset.js afterward instead (e.g. you
# hand-edit the pack later), that still works fine against whatever's in
# ascii_art_pack___small/ — the two tools don't need to agree on RNG, only
# on the folder format (10 rows x 20 cols, 2 symbols).
# ---------------------------------------------------------------------------

W, H = 20, 10
BG, FG = '.', '#'  # background/foreground symbols — vocab[0] is always BG (see textToBinary in js/similarity.js)

DATA_DIR = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "ascii_art_pack___small")  # was a bare top-level dir; build_dataset.js/generate_placeholder_pack.js both expect it under data/

DEFAULT_TOTAL_IMAGES = 20000
# ~150 examples per family across the 13 continuously-parameterized families
# below — dense enough for a much bigger local model (see ../python/train.py)
# to actually learn each family's shape well rather than a handful of
# instances of it, while still holding out the other ~90% (18,000 files) as
# a large pool of "shots on goal" for the SSIM search. Same reasoning as the
# smaller in-browser-friendly defaults this replaced (2,000 / 300, ~23/family)
# — just scaled with the model, since a bigger network with the old sparse
# coverage would mostly just learn to memorize individual training files
# instead of the family they belong to.
DEFAULT_TRAIN_SIZE = 2000
SEED = 0xC0FFEE  # same constant build_dataset.js uses, for the same reason: reproducible runs

TOTAL_IMAGES = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOTAL_IMAGES
TRAIN_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TRAIN_SIZE

# Third argument: "small" (default, 20x10 - what every gen_* function below
# is tuned for), "large" (60x30 - the scale the very first version of this
# project used, per README.md's "What changed in this version"), or any
# raw integer scale factor. Every shape is still generated in the exact
# same 20x10 reference coordinate space as always (CX, CY, every radius/
# size random.uniform/randint call below is completely untouched) and then
# block-upscaled - each reference character becomes an SxS block of the
# same character. This means a "large" ring is really a "small" ring's
# exact proportions blown up, not a re-tuned shape with its own new random
# ranges, which is the easiest way to add a size knob without touching (or
# risking a bug in) any of the per-family geometry below.
SIZE_PRESETS = {"small": 1, "large": 3}
_size_arg = sys.argv[3] if len(sys.argv) > 3 else "small"
SCALE = SIZE_PRESETS.get(_size_arg, None)
if SCALE is None:
    try:
        SCALE = int(_size_arg)
    except ValueError:
        raise SystemExit(f'unrecognized size "{_size_arg}" - use "small", "large", or an integer scale factor')
if SCALE < 1:
    raise SystemExit(f"scale must be >= 1, got {SCALE}")

OUT_W, OUT_H = W * SCALE, H * SCALE


def upscale_grid(text, scale):
    """Block-upscale a text grid by an integer factor: each character becomes
    an scale x scale block of that same character. scale=1 is a no-op."""
    if scale == 1:
        return text
    out_rows = []
    for row in text.split('\n'):
        wide_row = ''.join(ch * scale for ch in row)
        out_rows.extend([wide_row] * scale)
    return '\n'.join(out_rows)

CX, CY = W // 2, H // 2  # canvas center (10, 5)


def blank():
    return [[BG for _ in range(W)] for _ in range(H)]


def set_px(c, x, y, ch=FG):
    # The one place any generator touches the grid. Bounds-checked so a
    # shape that would extend past the canvas just clips at the edge
    # instead of crashing — this is the actual fix for the IndexError bug.
    xi, yi = int(round(x)), int(round(y))
    if 0 <= yi < H and 0 <= xi < W:
        c[yi][xi] = ch


def render(c):
    return '\n'.join(''.join(row) for row in c)


def jitter_center():
    # Small jitter around the true center so shapes aren't all dead-centered,
    # while staying close enough that few of them clip badly at 20x10.
    return random.randint(CX - 2, CX + 2), random.randint(CY - 1, CY + 1)


def gen_heart(cx=CX, cy=CY):
    c = blank()
    scale = random.uniform(2.0, 3.2)
    fill = random.choice([True, True, False])
    for y in range(H):
        for x in range(W):
            nx = (x - cx) / (scale * 1.8)
            ny = -(y - cy) / scale
            if nx ** 2 + ny ** 2 > 0:
                val = (nx ** 2 + ny ** 2 - 1) ** 3 - (nx ** 2) * (ny ** 3)
                if fill and val <= 0:
                    set_px(c, x, y)
                elif not fill and -0.18 <= val <= 0.02:
                    set_px(c, x, y)
    return c


def gen_circle(cx=CX, cy=CY):
    c = blank()
    r = random.uniform(2.0, 4.3)
    fill = random.choice([True, False])
    for y in range(H):
        for x in range(W):
            dx = (x - cx) / 1.8
            dy = (y - cy)
            dist = math.sqrt(dx * dx + dy * dy)
            if fill and dist <= r:
                set_px(c, x, y)
            elif not fill and abs(dist - r) < 0.6:
                set_px(c, x, y)
    return c


def gen_ring(cx=CX, cy=CY):
    c = blank()
    r_outer = random.uniform(3.2, 4.5)
    r_inner = r_outer * random.uniform(0.35, 0.65)
    for y in range(H):
        for x in range(W):
            dx = (x - cx) / 1.8
            dy = (y - cy)
            dist = math.sqrt(dx * dx + dy * dy)
            if r_inner <= dist <= r_outer:
                set_px(c, x, y)
    return c


def gen_diamond(cx=CX, cy=CY):
    c = blank()
    size = random.uniform(2.5, 4.3)
    fill = random.choice([True, False])
    for y in range(H):
        for x in range(W):
            dx = abs(x - cx) / 1.8
            dy = abs(y - cy)
            val = dx + dy
            if fill and val <= size:
                set_px(c, x, y)
            elif not fill and abs(val - size) < 0.6:
                set_px(c, x, y)
    return c


def gen_rect(cx=CX, cy=CY):
    c = blank()
    w_size = random.randint(3, 8)
    h_size = random.randint(2, 4)
    fill = random.choice([True, False])
    x1, x2 = max(0, cx - w_size), min(W - 1, cx + w_size)
    y1, y2 = max(0, cy - h_size), min(H - 1, cy + h_size)
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            if fill or y == y1 or y == y2 or x == x1 or x == x2:
                set_px(c, x, y)
    return c


def gen_triangle(cx=CX, cy=CY):
    c = blank()
    size = random.randint(2, 4)
    direction = random.choice(['up', 'down'])
    for y in range(H):
        for x in range(W):
            dx = abs(x - cx) / 1.8
            if direction == 'up':
                dy = cy + size - y
                if 0 <= dy <= size * 2 and dx <= dy / 2:
                    set_px(c, x, y)
            else:
                dy = y - (cy - size)
                if 0 <= dy <= size * 2 and dx <= (size * 2 - dy) / 2:
                    set_px(c, x, y)
    return c


def gen_star(cx=CX, cy=CY):
    c = blank()
    r_out = random.uniform(3.3, 4.5)
    r_in = r_out * random.uniform(0.3, 0.5)
    pts = random.choice([4, 5, 6, 8])
    for y in range(H):
        for x in range(W):
            dx = (x - cx) / 1.8
            dy = (y - cy)
            angle = math.atan2(dy, dx)
            dist = math.sqrt(dx * dx + dy * dy)
            star_r = r_in + (r_out - r_in) * (0.5 + 0.5 * math.cos(pts * angle))
            if dist <= star_r:
                set_px(c, x, y)
    return c


def gen_cross(cx=CX, cy=CY):
    c = blank()
    arm_w = random.randint(1, 2)
    arm_l = random.randint(3, 5)
    for y in range(H):
        for x in range(W):
            dx = abs(x - cx)
            dy = abs(y - cy)
            if (dx <= arm_w * 1.8 and dy <= arm_l) or (dy <= arm_w and dx <= arm_l * 1.8):
                set_px(c, x, y)
    return c


def gen_invader(cx=CX, cy=CY):
    c = blank()
    rows, cols_half = 6, 3
    grid = [[random.choice([0, 1]) for _ in range(cols_half)] for _ in range(rows)]
    for r in range(rows):
        full_row = grid[r] + grid[r][::-1]
        for col_idx, val in enumerate(full_row):
            if val:
                px = cx - (len(full_row) * 1.8) / 2 + col_idx * 1.8
                py = cy - rows / 2 + r
                set_px(c, px, py)
    return c


def gen_crescent(cx=CX, cy=CY):
    c = blank()
    r = random.uniform(2.8, 4.2)
    off_x = random.uniform(1.0, 1.6)
    off_y = random.uniform(-1.0, -0.4)
    for y in range(H):
        for x in range(W):
            dx1 = (x - cx) / 1.8
            dy1 = (y - cy)
            dist1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
            dx2 = (x - (cx + off_x)) / 1.8
            dy2 = (y - (cy + off_y))
            dist2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
            if dist1 <= r and dist2 >= r * 0.75:
                set_px(c, x, y)
    return c


def gen_tree(cx=CX, cy=CY):
    # Redesigned for 10 rows: a tapering triangular canopy over the top
    # rows, a 1-2 wide trunk in the bottom 2. The original version indexed
    # rows up to y=25 on this grid, which is the crash this rewrite fixes.
    c = blank()
    canopy_h = random.randint(5, 7)
    trunk_h = H - canopy_h
    for i in range(canopy_h):
        y = i
        half_w = max(1, round((i + 1) * (W / 2 - 1) / canopy_h / 1.8))
        for x in range(cx - half_w, cx + half_w + 1):
            set_px(c, x, y)
    trunk_w = random.choice([0, 1])
    for y in range(canopy_h, H):
        for x in range(cx - trunk_w, cx + trunk_w + 1):
            set_px(c, x, y)
    return c


def gen_house(cx=CX, cy=CY):
    # Also redesigned for 10 rows (same crash as gen_tree, same fix).
    c = blank()
    roof_h = 3
    body_h = H - roof_h
    body_half_w = random.randint(4, 7)
    for i in range(roof_h):
        y = i
        half_w = round((i + 1) * body_half_w / roof_h)
        for x in range(cx - half_w, cx + half_w + 1):
            set_px(c, x, y)
    x1, x2 = cx - body_half_w, cx + body_half_w
    y1, y2 = roof_h, H - 1
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            set_px(c, x, y)
    door_w = 1
    for y in range(y2 - 1, y2 + 1):
        for x in range(cx - door_w, cx + door_w + 1):
            set_px(c, x, y, BG)
    return c


def gen_target(cx=CX, cy=CY):
    c = blank()
    rings = random.randint(2, 4)
    spacing = 1.1
    for y in range(H):
        for x in range(W):
            dx = (x - cx) / 1.8
            dy = (y - cy)
            dist = math.sqrt(dx * dx + dy * dy)
            ring_idx = int(dist / spacing)
            if ring_idx < rings and ring_idx % 2 == 0:
                set_px(c, x, y)
    return c


generators = [
    gen_heart, gen_circle, gen_ring, gen_diamond, gen_rect,
    gen_triangle, gen_star, gen_cross, gen_invader, gen_crescent,
    gen_tree, gen_house, gen_target,
]


def main():
    random.seed(SEED)  # whole run is reproducible: same pack, same split, every time

    print(f"Generating {TOTAL_IMAGES} ASCII files ({OUT_W}x{OUT_H}, symbols '{BG}'/'{FG}', "
          f"scale {SCALE}x from a {W}x{H} reference) in '{OUTPUT_DIR}'...")

    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)  # clear stale files from a previous run with a different TOTAL_IMAGES
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entries = []
    for i in range(1, TOTAL_IMAGES + 1):
        gen_fn = random.choice(generators)
        cx, cy = jitter_center()
        c = gen_fn(cx=cx, cy=cy)
        text = upscale_grid(render(c), SCALE)

        rows = text.split('\n')
        assert len(rows) == OUT_H and all(len(r) == OUT_W for r in rows), \
            f"{gen_fn.__name__} produced a {len(rows)}-row grid, expected {OUT_H}"
        used = set(text.replace('\n', ''))
        assert used <= {BG, FG}, f"{gen_fn.__name__} used unexpected character(s): {used - {BG, FG}}"

        fname = f"ascii_{i:04d}.txt"
        with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(text + "\n")
        entries.append({"file": fname, "text": text})

    print(f"Wrote {len(entries)} files to '{OUTPUT_DIR}/'.")

    # Same reasoning as build_dataset.js's identical cap: protects against
    # TOTAL_IMAGES being turned down without a matching TRAIN_SIZE, which
    # would otherwise silently consume nearly the whole pack as "train".
    train_size = min(TRAIN_SIZE, len(entries) // 2, len(entries) - 1)
    shuffled = entries[:]
    random.shuffle(shuffled)
    train = sorted(shuffled[:train_size], key=lambda e: e["file"])
    test = sorted(shuffled[train_size:], key=lambda e: e["file"])

    os.makedirs(DATA_DIR, exist_ok=True)
    dataset = {
        "height": OUT_H,
        "width": OUT_W,
        "vocab": [BG, FG],  # vocab[0] is background by construction — see set_px()/BG above
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourceCount": len(entries),
        "trainSize": len(train),
        "train": train,  # the only images the model ever sees
        "test": test,    # held out — never trained on, the only valid match target
    }
    out_path = os.path.join(DATA_DIR, "dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    print(f"Split: {len(train)} train / {len(test)} held-out test (requested train size {TRAIN_SIZE})")
    print(f"Wrote '{out_path}'.")


if __name__ == "__main__":
    main()