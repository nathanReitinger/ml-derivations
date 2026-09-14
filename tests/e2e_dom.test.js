const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const dom = new JSDOM(html, { url: 'http://localhost/', pretendToBeVisual: true });
const { window } = dom;

// --- Canvas 2D stub: jsdom doesn't implement the Canvas API, and we don't
// need real pixels for this test (rendering correctness was already
// validated separately) — just enough to not throw. ---
const fakeCtx = {
  fillRect() {}, clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
  set fillStyle(_) {}, set strokeStyle(_) {}, set lineWidth(_) {},
};
window.HTMLCanvasElement.prototype.getContext = (kind) => (kind === '2d' ? fakeCtx : null);
// jsdom doesn't implement scrollIntoView (real browsers do) - stub it; this
// is pre-existing app.js behavior I didn't touch, not part of what I'm testing.
window.HTMLElement.prototype.scrollIntoView = () => {};

// --- globals app.js expects to find bare (as if loaded via <script> tags) ---
global.window = window;
global.document = window.document;
global.performance = performance;
global.URL = URL;
global.Blob = Blob;
global.devicePixelRatio = 1;
global.tf = require('@tensorflow/tfjs');
global.DiffusionModel = require('../js/model.js');
global.DiffusionData = require('../js/data.js');
// render.js/similarity.js use a simpler UMD variant that always attaches to
// `window` (or `this`) rather than checking module.exports, so pull them off
// window after requiring rather than trusting require()'s return value.
require('../js/render.js');
require('../js/similarity.js');
global.AsciiRender = window.AsciiRender;
global.AsciiSimilarity = window.AsciiSimilarity;
global.window.BUILD_ID = 'test';

// --- fetch stub: serve the tiny local dataset instead of a real network fetch ---
global.fetch = async (url) => {
  if (url.includes('dataset.json')) {
    const text = fs.readFileSync(path.join(__dirname, 'fixtures', 'dataset_tiny.json'), 'utf8');
    return { ok: true, json: async () => JSON.parse(text) };
  }
  if (url.includes('weights.json')) {
    return { ok: false, status: 404 }; // none committed in this test - exercises the fallback-to-train path
  }
  if (url.includes('showcase.json')) {
    return {
      ok: true,
      json: async () => ({
        generatedText: fs.readFileSync(path.join(__dirname, 'fixtures', 'dataset_tiny.json'), 'utf8')
          && JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'dataset_tiny.json'), 'utf8')).test[0].text,
        heldOutText: JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'dataset_tiny.json'), 'utf8')).test[0].text,
        heldOutFile: 'test_0.txt', ssim: 0.987, hamming: 1.0, attempts: 4242, elapsedSeconds: 12.3,
        modelStep: 9999, createdAt: '2026-09-14T00:00:00Z',
      }),
    };
  }
  throw new Error('unexpected fetch: ' + url);
};

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
async function waitUntil(fn, timeoutMs = 20000, stepMs = 20) {
  const t0 = Date.now();
  while (!fn()) {
    if (Date.now() - t0 > timeoutMs) throw new Error('waitUntil timed out');
    await sleep(stepMs);
  }
}

async function main() {
  // Loosen the safety-pause ceiling isn't needed; just drive tiny hyperparams.
  require('../js/app.js'); // IIFE runs immediately, calls init() at the end

  const $ = (id) => window.document.getElementById(id);

  await waitUntil(() => $('btn-train').disabled === false, 10000);
  console.log('1. dataset loaded, Train button enabled. backend:', $('stat-status').textContent);
  await waitUntil(() => !$('pretrained-status').textContent.includes('Checking'), 10000, 20);
  console.log('   no weights.json committed -> defaulted to train mode:',
    $('mode-train-content').hidden === false, '| existing content hidden:', $('mode-existing-content').hidden === true);
  console.log('   pretrained-status text:', $('pretrained-status').textContent);
  console.log('0. showcase panel rendered from stubbed showcase.json:', $('showcase-panel').hidden === false);
  console.log('   showcase score line:', $('showcase-score').textContent);

  // Tiny, fast hyperparameters so the quality gate passes in a handful of steps.
  // (This sandbox has no WebGL/native backend, so tf.js falls back to its pure-JS
  // cpu backend, which is ~100x+ slower than a real browser's WebGL backend or
  // tfjs-node's native binding - fine for a correctness check, not a speed one.)
  $('hp-hidden').value = '6';
  $('hp-timesteps').value = '4';
  $('hp-steps').value = '15';
  $('hp-batch').value = '6';
  $('hp-eval-every').value = '5';
  $('hp-eval-samples').value = '2';
  $('hp-train-target').value = '0.01';   // deliberately trivial - this is a plumbing test, not a quality test
  $('hp-threshold').value = '0.01';       // ditto, for generation matching

  $('btn-train').dispatchEvent(new window.Event('click'));
  await waitUntil(() => $('btn-train').disabled === false || $('btn-train-more').hidden === false, 120000, 100);
  console.log('2. training finished:', $('train-status').textContent);
  console.log('   generate enabled:', $('btn-generate').disabled === false);

  if ($('btn-generate').disabled) {
    console.log('   (quality target not hit yet - forcing "train more" once)');
    $('btn-train-more').dispatchEvent(new window.Event('click'));
    await waitUntil(() => $('btn-generate').disabled === false, 120000, 100);
  }

  $('btn-generate').dispatchEvent(new window.Event('click'));
  await waitUntil(() => $('match-panel').hidden === false, 120000, 100);
  console.log('3. match found:', $('match-meta').textContent);
  console.log('   match score:', $('match-score').textContent);
  console.log('   filmstrip tiles:', $('filmstrip').children.length);

  // --- test handleLoadWeightsFile via a fake File-like object ---
  const tinyModel = global.DiffusionModel.buildModel({
    height: 10, width: 20, vocabSize: 2, hidden: 8, timeDim: 16, bottleneckBlocks: 2, bottleneckKernel: 3,
  });
  tinyModel.warmUp();
  function dumpLayer(layer) {
    const out = {};
    for (const w of layer.getWeights()) {
      const key = w.name.includes('kernel') ? 'kernel' : w.name.includes('bias') ? 'bias'
                : w.name.includes('gamma') ? 'gamma' : w.name.includes('beta') ? 'beta' : w.name;
      out[key] = { shape: w.shape, data: Array.from(w.dataSync()) };
    }
    return out;
  }
  const w = {};
  for (const [name, layer] of Object.entries(tinyModel.layers)) {
    if (name === 'bottleneck') {
      w.bottleneck = layer.map((block) => {
        const b = {};
        for (const [sub, l] of Object.entries(block)) b[sub] = dumpLayer(l);
        return b;
      });
    } else if (layer && layer.getWeights) {
      w[name] = dumpLayer(layer);
    }
  }
  const bundle = {
    meta: { height: 10, width: 20, vocabSize: 2, hidden: 8, timeDim: 16, bottleneckBlocks: 2, bottleneckKernel: 3,
             timesteps: 6, step: 1234, meanQuality: 0.5 },
    weights: w,
  };
  const fakeFile = { name: 'weights.json', text: async () => JSON.stringify(bundle) };
  const fileInput = $('load-weights-file');
  Object.defineProperty(fileInput, 'files', { value: [fakeFile], configurable: true });
  fileInput.dispatchEvent(new window.Event('change'));
  await waitUntil(() => $('pretrained-status').textContent.includes('Loaded'), 10000, 20);
  console.log('4. load-pretrained-weights status:', $('pretrained-status').textContent);
  console.log('   generate enabled after load:', $('btn-generate').disabled === false);
  console.log('   mode switched to "existing":', $('mode-existing-content').hidden === false,
    '| train content hidden:', $('mode-train-content').hidden === true);

  console.log('\nALL STEPS COMPLETED WITHOUT UNCAUGHT ERRORS');
}

main().catch((err) => {
  console.error('TEST FAILED:', err);
  process.exit(1);
});
