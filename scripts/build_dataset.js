#!/usr/bin/env node
// Builds data/dataset.json from the .txt files in data/ascii_art_pack___small.
//
// Every file must be exactly HEIGHT rows of WIDTH characters, using exactly
// two distinct symbols (the model, the similarity metric, and the whole
// premise of this site assume a 2-symbol vocabulary).
//
// The split is train vs. held-out test, same idea as the old large-pack
// version, just resized: a fixed-size TRAIN_SIZE subset (50 files by
// default, deterministically chosen) is what the model ever trains on.
// Everything else becomes "test" — held out, never shown to the model
// during training, and it's the ONLY thing the generation-side matching in
// js/app.js is allowed to compare against. That boundary is the entire
// point of the site (a small enough, repetitive-enough problem space lets a
// model land close to data it never trained on), so nothing in the training
// loop — including the quality gate — may read from `test`. Only `train`.
//
// Usage: node scripts/build_dataset.js [sourceDir] [outFile] [trainSize]

const fs = require('fs');
const path = require('path');

// HEIGHT/WIDTH are no longer hardcoded here - inferred from the pack itself
// (see main()) so this same script validates either the 20x10 "small" pack
// or a 60x30 (or any other size) "large" pack g.py can also produce,
// without needing a flag to say which.
const SPLIT_SEED = 0xC0FFEE; // fixed on purpose — see note below
const DEFAULT_TRAIN_SIZE = 2000;

const SRC_DIR = process.argv[2] || path.join(__dirname, '..', 'data', 'ascii_art_pack___small');
const OUT_FILE = process.argv[3] || path.join(__dirname, '..', 'data', 'dataset.json');
const TRAIN_SIZE = process.argv[4] ? parseInt(process.argv[4], 10) : DEFAULT_TRAIN_SIZE;

// Deterministic PRNG (mulberry32) so the train/test split is reproducible
// across machines/CI, as long as the same set of source files is used.
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle(arr, seed) {
  const rand = mulberry32(seed);
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Must match the normalization used client-side (js/data.js) exactly, or an
// exact character-grid comparison could silently misbehave due to a
// whitespace mismatch.
function normalizeText(text) {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').replace(/\n+$/, '');
}

function main() {
  if (!fs.existsSync(SRC_DIR)) {
    console.error(`Source directory not found: ${SRC_DIR}`);
    console.error('Tip: node scripts/generate_placeholder_pack.js will create a small placeholder pack here.');
    process.exit(1);
  }

  const allFiles = fs.readdirSync(SRC_DIR)
    .filter(f => f.toLowerCase().endsWith('.txt'))
    .sort(); // sort first so the shuffle below is deterministic regardless of OS directory order

  if (allFiles.length === 0) {
    console.error(`No .txt files found in ${SRC_DIR}`);
    process.exit(1);
  }

  // Infer the expected grid size from the first file that actually parses
  // into a rectangular grid (rather than assuming 20x10) - this is what
  // lets the exact same script validate a "small" (20x10) or "large"
  // (60x30, or anything else g.py was asked for) pack with no flag needed.
  let HEIGHT = null, WIDTH = null;
  for (const file of allFiles) {
    const rows = normalizeText(fs.readFileSync(path.join(SRC_DIR, file), 'utf8')).split('\n');
    if (rows.length > 0 && rows[0].length > 0) {
      HEIGHT = rows.length;
      WIDTH = rows[0].length;
      break;
    }
  }
  if (HEIGHT === null) {
    console.error(`Could not infer a grid size - every file in ${SRC_DIR} appears empty.`);
    process.exit(1);
  }
  console.log(`Inferred grid size ${WIDTH}x${HEIGHT} from ${allFiles[0]}`);

  const valid = [];
  const skipped = [];
  const charSet = new Set();

  for (const file of allFiles) {
    const raw = fs.readFileSync(path.join(SRC_DIR, file), 'utf8');
    const text = normalizeText(raw);
    const rows = text.split('\n');
    if (rows.length !== HEIGHT || rows.some(r => r.length !== WIDTH)) {
      skipped.push({ file, reason: `expected ${HEIGHT}x${WIDTH}, got ${rows.length} rows` });
      continue;
    }
    for (const ch of text) if (ch !== '\n') charSet.add(ch);
    valid.push({ file, text });
  }

  if (valid.length < 2) {
    console.error(`Only ${valid.length} valid ${HEIGHT}x${WIDTH} file(s) found in ${SRC_DIR}. Need at least 2.`);
    process.exit(1);
  }

  const vocab = Array.from(charSet).sort();
  if (vocab.length !== 2) {
    console.warn(
      `WARNING: found ${vocab.length} distinct characters (${vocab.map(c => JSON.stringify(c)).join(', ')}), ` +
      `not 2. The model, the UI copy, and the similarity metric all assume a 2-symbol vocabulary — ` +
      `things will still run, but "close match" scoring will treat every non-first vocab char as one ` +
      `undifferentiated "foreground" symbol.`
    );
  }

  // min() with valid.length/2 as well as -1: on a big pack (g.py's 20k
  // default) this never binds and TRAIN_SIZE (2000) is used as requested.
  // On a small pack (e.g. the 66-file placeholder), TRAIN_SIZE=2000 would
  // otherwise consume nearly the whole thing, leaving almost nothing held
  // out — capping at half keeps the split sane automatically instead of
  // requiring an explicit trainSize argument for every small pack.
  const trainSize = Math.min(TRAIN_SIZE, Math.floor(valid.length / 2), valid.length - 1);
  const shuffled = seededShuffle(valid, SPLIT_SEED);
  const train = shuffled.slice(0, trainSize).sort((a, b) => a.file.localeCompare(b.file));
  const test = shuffled.slice(trainSize).sort((a, b) => a.file.localeCompare(b.file));

  const dataset = {
    height: HEIGHT,
    width: WIDTH,
    vocab,
    generatedAt: new Date().toISOString(),
    sourceCount: allFiles.length,
    skippedCount: skipped.length,
    trainSize,
    train, // the only images the model ever sees, and the only thing the quality gate may check against
    test,  // held out — never trained on, and the only valid match target during generation
  };

  fs.mkdirSync(path.dirname(OUT_FILE), { recursive: true });
  fs.writeFileSync(OUT_FILE, JSON.stringify(dataset));

  console.log(`Scanned ${allFiles.length} .txt file(s) in ${SRC_DIR}`);
  if (skipped.length) {
    console.log(`Skipped ${skipped.length} file(s) with the wrong dimensions:`);
    for (const s of skipped.slice(0, 20)) console.log(`  - ${s.file}: ${s.reason}`);
    if (skipped.length > 20) console.log(`  ...and ${skipped.length - 20} more`);
  }
  console.log(`Vocabulary (${vocab.length} chars): ${vocab.map(c => JSON.stringify(c)).join(', ')}`);
  console.log(`Split: ${train.length} train / ${test.length} held-out test (requested train size ${TRAIN_SIZE})`);
  console.log(`Wrote ${OUT_FILE}`);
}

main();
