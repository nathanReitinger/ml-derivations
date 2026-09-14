const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const html = fs.readFileSync(path.join(__dirname, '..', 'index.html'), 'utf8');
const dom = new JSDOM(html, { url: 'http://localhost/', pretendToBeVisual: true });
const { window } = dom;

const fakeCtx = {
  fillRect() {}, clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
  set fillStyle(_) {}, set strokeStyle(_) {}, set lineWidth(_) {},
};
window.HTMLCanvasElement.prototype.getContext = (kind) => (kind === '2d' ? fakeCtx : null);
window.HTMLElement.prototype.scrollIntoView = () => {};

global.window = window;
global.document = window.document;
global.performance = performance;
global.URL = URL;
global.Blob = Blob;
global.tf = require('@tensorflow/tfjs');
global.DiffusionModel = require('../js/model.js');
global.DiffusionData = require('../js/data.js');
require('../js/render.js');
require('../js/similarity.js');
global.AsciiRender = window.AsciiRender;
global.AsciiSimilarity = window.AsciiSimilarity;
global.window.BUILD_ID = 'test';

const datasetText = fs.readFileSync(path.join(__dirname, 'fixtures', 'dataset_tiny.json'), 'utf8');
const dataset = JSON.parse(datasetText);

// Build a small pretrained-style bundle up front (same shape bridge.py produces).
const tinyModel = global.DiffusionModel.buildModel({
  height: dataset.height, width: dataset.width, vocabSize: dataset.vocab.length,
  hidden: 8, timeDim: 16, bottleneckBlocks: 2, bottleneckKernel: 3,
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
const weightsBundle = {
  meta: { height: dataset.height, width: dataset.width, vocabSize: dataset.vocab.length,
           hidden: 8, timeDim: 16, bottleneckBlocks: 2, bottleneckKernel: 3,
           timesteps: 6, step: 5555, meanQuality: 0.93 },
  weights: w,
};

// This time, data/weights.json IS "committed" - the fetch stub serves it from the start.
let weightsJsonFetchCount = 0;
global.fetch = async (url) => {
  if (url.includes('dataset.json')) return { ok: true, json: async () => dataset };
  if (url.includes('weights.json')) { weightsJsonFetchCount++; return { ok: true, json: async () => weightsBundle }; }
  return { ok: false, status: 404 }; // showcase.json etc. - fine to be absent
};

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
async function waitUntil(fn, timeoutMs = 10000, stepMs = 20) {
  const t0 = Date.now();
  while (!fn()) {
    if (Date.now() - t0 > timeoutMs) throw new Error('waitUntil timed out');
    await sleep(stepMs);
  }
}

async function main() {
  require('../js/app.js');
  const $ = (id) => window.document.getElementById(id);

  // Diagnostic: print progress every 500ms instead of waiting blind for 10s.
  for (let i = 0; i < 20; i++) {
    if (!$('pretrained-status').textContent.includes('Checking')) break;
    await sleep(500);
  }

  const modeIsExisting = $('mode-existing-content').hidden === false && $('mode-train-content').hidden === true;
  const generateReady = $('btn-generate').disabled === false;
  const statusMentionsParams = $('pretrained-status').textContent.includes('params');

  console.log('pretrained-status:', $('pretrained-status').textContent);
  console.log('defaulted to "existing" mode:', modeIsExisting);
  console.log('Generate button enabled with no click needed:', generateReady);

  // Switching to "Train your own" afterward must not leave Train permanently
  // disabled (applyWeightsBundle() disables it while in "existing" mode).
  $('btn-mode-train').dispatchEvent(new window.Event('click'));
  const trainReenabled = $('btn-train').disabled === false;
  const modeNowTrain = $('mode-train-content').hidden === false && $('mode-existing-content').hidden === true;
  console.log('switched to "Train your own" -> Train button re-enabled:', trainReenabled, '| mode shown:', modeNowTrain);

  // Clicking back to "Use existing model" without having actually trained
  // anything should NOT re-fetch (the pretrained model is still the active
  // one - only trainModel() itself invalidates that) - it's still 1 so far.
  const fetchCountBeforeClick = weightsJsonFetchCount;
  $('btn-mode-existing').dispatchEvent(new window.Event('click'));
  await sleep(50);
  const noRedundantFetch = weightsJsonFetchCount === fetchCountBeforeClick;
  const backToExisting = $('mode-existing-content').hidden === false;
  console.log('clicked back to "Use existing model" (nothing was trained) -> no redundant fetch:',
    noRedundantFetch, `(count stayed at ${weightsJsonFetchCount})`, '| showing existing panel:', backToExisting);

  if (modeIsExisting && generateReady && statusMentionsParams && trainReenabled && modeNowTrain
      && noRedundantFetch && backToExisting) {
    console.log('PASS - auto-load works, mode switching recovers correctly, and no wasted re-fetches');
  } else {
    console.log('FAIL');
    process.exit(1);
  }
}

main().catch((err) => { console.error('TEST FAILED:', err); process.exit(1); });
