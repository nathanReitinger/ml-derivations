#!/usr/bin/env node
// Generates a small placeholder pack of 20x10, 2-symbol ascii drawings into
// data/ascii_art_pack___small. This is ONLY here so the site works out of
// the box before you swap in your own pack — same idea as the old large
// pack's placeholder. Every family here is deliberately repetitive/
// parametric (boxes, rings, stripes, crosses...) rather than one-of-a-kind,
// because that's what a tiny model trained on ~50 examples can actually
// learn to reproduce closely — see README "What to actually expect".
//
// Usage: node scripts/generate_placeholder_pack.js [outDir]

const fs = require('fs');
const path = require('path');

const WIDTH = 20;
const HEIGHT = 10;
const FG = '#';
const BG = '.';

const OUT_DIR = process.argv[2] || path.join(__dirname, '..', 'data', 'ascii_art_pack___small');

function blank() {
  return Array.from({ length: HEIGHT }, () => Array.from({ length: WIDTH }, () => BG));
}

function toText(grid) {
  return grid.map(row => row.join('')).join('\n');
}

function setPx(grid, x, y, ch = FG) {
  if (y >= 0 && y < HEIGHT && x >= 0 && x < WIDTH) grid[y][x] = ch;
}

const files = []; // { name, text }

function add(name, grid) {
  files.push({ name, text: toText(grid) });
}

// ---- hollow rectangles --------------------------------------------------
[[2, 2, 15, 6], [0, 0, 19, 9], [4, 3, 11, 4], [1, 1, 8, 5], [6, 1, 13, 8], [3, 0, 16, 3]]
  .forEach(([x0, y0, x1, y1], i) => {
    const g = blank();
    for (let x = x0; x <= x1; x++) { setPx(g, x, y0); setPx(g, x, y1); }
    for (let y = y0; y <= y1; y++) { setPx(g, x0, y); setPx(g, x1, y); }
    add(`box_${i}_${x0}${y0}${x1}${y1}`, g);
  });

// ---- filled rectangles ---------------------------------------------------
[[2, 2, 9, 6], [10, 1, 17, 8], [0, 3, 6, 6], [8, 0, 19, 2], [3, 6, 14, 9]]
  .forEach(([x0, y0, x1, y1], i) => {
    const g = blank();
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) setPx(g, x, y);
    add(`filledbox_${i}_${x0}${y0}${x1}${y1}`, g);
  });

// ---- diamond rings (Manhattan-distance bands from a center) -------------
[[9, 4, 3], [9, 4, 5], [9, 4, 7], [5, 4, 4], [13, 5, 4], [9, 4, 2], [9, 4, 6]]
  .forEach(([cx, cy, r], i) => {
    const g = blank();
    for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) {
      if (Math.abs(x - cx) + Math.abs(y - cy) === r) setPx(g, x, y);
    }
    add(`ring_${i}_${cx}${cy}${r}`, g);
  });

// ---- filled diamonds ------------------------------------------------------
[[9, 4, 3], [9, 4, 5], [5, 4, 3], [13, 4, 3]]
  .forEach(([cx, cy, r], i) => {
    const g = blank();
    for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) {
      if (Math.abs(x - cx) + Math.abs(y - cy) <= r) setPx(g, x, y);
    }
    add(`diamond_${i}_${cx}${cy}${r}`, g);
  });

// ---- crosses / plus signs -------------------------------------------------
[[9, 4, 1], [9, 4, 2], [9, 4, 3], [5, 4, 2], [13, 4, 2], [9, 2, 2], [9, 6, 2]]
  .forEach(([cx, cy, t], i) => {
    const g = blank();
    for (let x = 0; x < WIDTH; x++) for (let dy = -t; dy <= t; dy++) setPx(g, x, cy + dy);
    for (let y = 0; y < HEIGHT; y++) for (let dx = -t; dx <= t; dx++) setPx(g, cx + dx, y);
    add(`cross_${i}_${cx}${cy}${t}`, g);
  });

// ---- diagonal X marks ------------------------------------------------------
[0, 1, 2].forEach((t, i) => {
  const g = blank();
  for (let k = 0; k < Math.max(WIDTH, HEIGHT); k++) {
    for (let dt = -t; dt <= t; dt++) {
      setPx(g, k, Math.round(k * (HEIGHT - 1) / (WIDTH - 1)) + dt);
      setPx(g, k, HEIGHT - 1 - Math.round(k * (HEIGHT - 1) / (WIDTH - 1)) + dt);
    }
  }
  add(`xmark_${i}_${t}`, g);
});

// ---- horizontal stripes ----------------------------------------------------
[2, 3, 4, 5].forEach((period, i) => {
  const g = blank();
  for (let y = 0; y < HEIGHT; y++) if (y % period < Math.ceil(period / 2)) {
    for (let x = 0; x < WIDTH; x++) setPx(g, x, y);
  }
  add(`hstripes_${i}_${period}`, g);
});

// ---- vertical stripes -------------------------------------------------------
[2, 3, 4, 5].forEach((period, i) => {
  const g = blank();
  for (let x = 0; x < WIDTH; x++) if (x % period < Math.ceil(period / 2)) {
    for (let y = 0; y < HEIGHT; y++) setPx(g, x, y);
  }
  add(`vstripes_${i}_${period}`, g);
});

// ---- diagonal stripes ---------------------------------------------------------
[3, 4, 5, 6].forEach((period, i) => {
  const g = blank();
  for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) {
    if ((x + y) % period < Math.ceil(period / 2)) setPx(g, x, y);
  }
  add(`diagstripes_${i}_${period}`, g);
});

// ---- checkerboards ---------------------------------------------------------
[1, 2, 3].forEach((size, i) => {
  const g = blank();
  for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) {
    if ((Math.floor(x / size) + Math.floor(y / size)) % 2 === 0) setPx(g, x, y);
  }
  add(`checker_${i}_${size}`, g);
});

// ---- triangles (pointing in each direction, a couple of sizes) -------------
function triUp(g, cx, baseY, h) {
  for (let dy = 0; dy < h; dy++) {
    const y = baseY - dy;
    for (let dx = -dy; dx <= dy; dx++) setPx(g, cx + dx, y);
  }
}
function triDown(g, cx, topY, h) {
  for (let dy = 0; dy < h; dy++) {
    const y = topY + dy;
    for (let dx = -(h - 1 - dy); dx <= (h - 1 - dy); dx++) setPx(g, cx + dx, y);
  }
}
[[9, 9, 6], [9, 9, 4]].forEach(([cx, baseY, h], i) => { const g = blank(); triUp(g, cx, baseY, h); add(`tri_up_${i}_${cx}${baseY}${h}`, g); });
[[9, 0, 6], [9, 0, 4]].forEach(([cx, topY, h], i) => { const g = blank(); triDown(g, cx, topY, h); add(`tri_down_${i}_${cx}${topY}${h}`, g); });

// ---- frames (border of a given thickness) ----------------------------------
[1, 2, 3].forEach((t, i) => {
  const g = blank();
  for (let y = 0; y < HEIGHT; y++) for (let x = 0; x < WIDTH; x++) {
    if (x < t || y < t || x >= WIDTH - t || y >= HEIGHT - t) setPx(g, x, y);
  }
  add(`frame_${i}_${t}`, g);
});

// ---- dot grids ----------------------------------------------------------------
[2, 3, 4].forEach((spacing, i) => {
  const g = blank();
  for (let y = 0; y < HEIGHT; y += spacing) for (let x = 0; x < WIDTH; x += spacing) setPx(g, x, y);
  add(`dotgrid_${i}_${spacing}`, g);
});

// ---- sine waves ------------------------------------------------------------------
[1, 2, 3].forEach((freq, i) => {
  const g = blank();
  for (let x = 0; x < WIDTH; x++) {
    const y = Math.round((HEIGHT - 1) / 2 + Math.sin((x / WIDTH) * Math.PI * 2 * freq) * (HEIGHT / 2 - 1));
    setPx(g, x, y);
    setPx(g, x, y + 1);
  }
  add(`wave_${i}_${freq}`, g);
});

// ---- arrows (left/right/up/down) --------------------------------------------------
function arrowRight(g) {
  const cy = 4;
  for (let x = 2; x <= 14; x++) setPx(g, x, cy);
  for (let d = 0; d < 5; d++) { setPx(g, 14 - d, cy - d - 1); setPx(g, 14 - d, cy + d + 1); }
}
function arrowLeft(g) {
  const cy = 4;
  for (let x = 5; x <= 17; x++) setPx(g, x, cy);
  for (let d = 0; d < 5; d++) { setPx(g, 5 + d, cy - d - 1); setPx(g, 5 + d, cy + d + 1); }
}
add('arrow_right_0', (() => { const g = blank(); arrowRight(g); return g; })());
add('arrow_left_0', (() => { const g = blank(); arrowLeft(g); return g; })());

// ---- L-shapes / corners ------------------------------------------------------------
[[0, 0], [19, 0], [0, 9], [19, 9]].forEach(([cx, cy], i) => {
  const g = blank();
  const dx = cx === 0 ? 1 : -1;
  const dy = cy === 0 ? 1 : -1;
  for (let k = 0; k < 10; k++) setPx(g, cx + dx * k, cy);
  for (let k = 0; k < 6; k++) setPx(g, cx, cy + dy * k);
  add(`corner_${i}_${cx}${cy}`, g);
});

// Validate + write.
fs.mkdirSync(OUT_DIR, { recursive: true });
let written = 0;
for (const { name, text } of files) {
  const rows = text.split('\n');
  if (rows.length !== HEIGHT || rows.some(r => r.length !== WIDTH)) {
    throw new Error(`generated shape "${name}" is malformed (${rows.length} rows)`);
  }
  const chars = new Set(text.replace(/\n/g, ''));
  for (const ch of chars) {
    if (ch !== FG && ch !== BG) throw new Error(`generated shape "${name}" used unexpected char "${ch}"`);
  }
  const fname = `${String(written).padStart(4, '0')}_${name}.txt`;
  fs.writeFileSync(path.join(OUT_DIR, fname), text + '\n');
  written++;
}

console.log(`Wrote ${written} placeholder files (${WIDTH}x${HEIGHT}, symbols "${FG}"/"${BG}") to ${OUT_DIR}`);
