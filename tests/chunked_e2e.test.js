const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');
const { JSDOM } = require('jsdom');

// Self-contained: generate this test's own fixture via the REAL bridge.py,
// through lightweight fake-torch stubs (no torch install needed) - rather
// than depending on a pre-existing file, so this test runs the same way
// anywhere `npm test` does.
const CHUNKED_DIR = fs.mkdtempSync(path.join(os.tmpdir(), 'chunked_e2e_'));
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
global.AsciiRender = window.AsciiRender;
global.AsciiSimilarity = window.AsciiSimilarity;
global.window.BUILD_ID = 'test';

const dataset = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'dataset_tiny.json'), 'utf8'));

let chunkFetchCount = 0;
global.fetch = async (url) => {
  if (url.includes('dataset.json')) return { ok: true, json: async () => dataset };
  if (url.endsWith('weights/manifest.json')) {
    const text = fs.readFileSync(path.join(CHUNKED_DIR, 'manifest.json'), 'utf8');
    return { ok: true, json: async () => JSON.parse(text) };
  }
  if (url.includes('weights/part-')) {
    chunkFetchCount++;
    const name = url.split('/').pop();
    return { ok: true, text: async () => fs.readFileSync(path.join(CHUNKED_DIR, name), 'utf8') };
  }
  return { ok: false, status: 404 }; // legacy weights.json, showcase.json - fine to be absent
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

  // btn-train is intentionally left disabled once a pretrained model loads
  // (see setMode) - wait on pretrained-status settling instead, same fix as
  // autoload_pretrained.test.js needed.
  await waitUntil(() => !$('pretrained-status').textContent.includes('Checking'), 10000, 20);

  const manifest = JSON.parse(fs.readFileSync(path.join(CHUNKED_DIR, 'manifest.json'), 'utf8'));
  const modeIsExisting = $('mode-existing-content').hidden === false && $('mode-train-content').hidden === true;
  const generateReady = $('btn-generate').disabled === false;
  const status = $('pretrained-status').textContent;

  console.log('manifest says', manifest.numChunks, 'chunks; fetch stub actually served', chunkFetchCount);
  console.log('pretrained-status:', status);
  console.log('defaulted to "existing" mode:', modeIsExisting);
  console.log('Generate enabled:', generateReady);

  const allChunksFetched = chunkFetchCount === manifest.numChunks;
  const statusMentionsChunked = status.includes('chunked');
  const statusMentionsQuality = status.includes('91.0%'); // meanQuality: 0.91 from the export step above

  if (modeIsExisting && generateReady && allChunksFetched && statusMentionsChunked && statusMentionsQuality) {
    console.log('PASS - a real chunked bridge.py export loads correctly via the real fetch/reassemble/loadWeights path');
  } else {
    console.log('FAIL');
    process.exit(1);
  }
  fs.rmSync(CHUNKED_DIR, { recursive: true, force: true });
}

main().catch((err) => { console.error('TEST FAILED:', err); process.exit(1); });
