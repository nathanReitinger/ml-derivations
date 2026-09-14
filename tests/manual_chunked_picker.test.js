const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const { JSDOM } = require('jsdom');

// Reuse the same real-bridge.py fixture generator as chunked_e2e.test.js.
const CHUNKED_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'manual_chunked_'));
execFileSync('python3', [path.join(__dirname, 'fixtures', 'dump_chunked_fixture.py'), CHUNKED_DIR]);

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
global.AsciiSimilarity = window.AsciiSimilarity;
global.AsciiRender = window.AsciiRender;
global.window.BUILD_ID = 'test';

const dataset = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'dataset_tiny.json'), 'utf8'));
global.fetch = async (url) => {
  if (url.includes('dataset.json')) return { ok: true, json: async () => dataset };
  return { ok: false, status: 404 }; // no committed weights - force "Train your own" default, we're testing the picker
};

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
async function waitUntil(fn, timeoutMs = 10000, stepMs = 20) {
  const t0 = Date.now();
  while (!fn()) {
    if (Date.now() - t0 > timeoutMs) throw new Error('waitUntil timed out');
    await sleep(stepMs);
  }
}

function fakeFileFrom(diskPath) {
  const name = path.basename(diskPath);
  return { name, text: async () => fs.readFileSync(diskPath, 'utf8') };
}

async function main() {
  require('../js/app.js');
  const $ = (id) => window.document.getElementById(id);
  const fileInput = $('load-weights-file');

  await waitUntil(() => $('btn-train').disabled === false, 10000);
  await waitUntil(() => !$('pretrained-status').textContent.includes('Checking'), 10000, 20);

  // --- Reproduce the exact bug report: selecting manifest.json ALONE ---
  const manifestOnly = [fakeFileFrom(path.join(CHUNKED_DIR, 'manifest.json'))];
  Object.defineProperty(fileInput, 'files', { value: manifestOnly, configurable: true });
  fileInput.dispatchEvent(new window.Event('change'));
  await waitUntil(() => $('pretrained-status').textContent.includes('Could not load'), 10000, 20);
  const errorMsg = $('pretrained-status').textContent;
  console.log('1. manifest.json selected alone -> error message:', errorMsg);
  const errorIsActionable = errorMsg.includes('part file') && errorMsg.includes('same file dialog');

  // --- The actual fix: manifest.json + all its real part files, selected together ---
  const manifest = JSON.parse(fs.readFileSync(path.join(CHUNKED_DIR, 'manifest.json'), 'utf8'));
  const allFiles = ['manifest.json', ...manifest.chunkFiles].map((name) => fakeFileFrom(path.join(CHUNKED_DIR, name)));
  Object.defineProperty(fileInput, 'files', { value: allFiles, configurable: true });
  fileInput.dispatchEvent(new window.Event('change'));
  await waitUntil(() => $('pretrained-status').textContent.includes('Loaded'), 10000, 20);
  const status = $('pretrained-status').textContent;
  const modeIsExisting = $('mode-existing-content').hidden === false;
  const generateReady = $('btn-generate').disabled === false;
  console.log('2. manifest.json + all parts selected together -> status:', status);
  console.log('   mode is "existing":', modeIsExisting, '| Generate enabled:', generateReady);

  fs.rmSync(CHUNKED_DIR, { recursive: true, force: true });

  if (errorIsActionable && modeIsExisting && generateReady && status.includes('params')) {
    console.log('PASS - clear error when incomplete, works correctly when manifest+parts selected together');
  } else {
    console.log('FAIL');
    process.exit(1);
  }
}

main().catch((err) => { console.error('TEST FAILED:', err); process.exit(1); });
