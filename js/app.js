(function () {
  const $ = (id) => document.getElementById(id);

  const els = {
    backend: $('stat-status'),
    buildBadge: $('build-badge'),
    statTrain: $('stat-train'),
    statTest: $('stat-test'),
    statVocab: $('stat-vocab'),
    statGrid: $('stat-grid'),
    trainGallery: $('train-gallery'),
    testGallery: $('test-gallery'),
    backendNote: $('backend-note'),

    btnTrain: $('btn-train'),
    btnTrainMore: $('btn-train-more'),
    btnGenerate: $('btn-generate'),
    btnStop: $('btn-stop'),
    loadWeightsFile: $('load-weights-file'),
    pretrainedStatus: $('pretrained-status'),
    btnModeExisting: $('btn-mode-existing'),
    btnModeTrain: $('btn-mode-train'),
    modeExistingContent: $('mode-existing-content'),
    modeTrainContent: $('mode-train-content'),

    hpSteps: $('hp-steps'),
    hpBatch: $('hp-batch'),
    hpHidden: $('hp-hidden'),
    hpT: $('hp-timesteps'),
    hpLr: $('hp-lr'),
    hpTrainTarget: $('hp-train-target'),
    hpEvalEvery: $('hp-eval-every'),
    hpEvalSamples: $('hp-eval-samples'),
    hpThreshold: $('hp-threshold'),

    trainProgress: $('train-progress'),
    trainStatus: $('train-status'),
    lossCanvas: $('loss-canvas'),
    qualityStatus: $('quality-status'),
    trainPreview: $('train-preview'),

    filmstrip: $('filmstrip'),
    genStatus: $('gen-status'),
    bestStatus: $('best-status'),

    matchPanel: $('match-panel'),
    matchMeta: $('match-meta'),
    matchGenCanvas: $('match-generated-canvas'),
    matchTestCanvas: $('match-test-canvas'),
    matchFilename: $('match-filename'),
    matchScore: $('match-score'),
    btnDownloadMatch: $('btn-download-match'),
    showcasePanel: $('showcase-panel'),
    showcaseMeta: $('showcase-meta'),
    showcaseGenCanvas: $('showcase-generated-canvas'),
    showcaseTestCanvas: $('showcase-test-canvas'),
    showcaseFilename: $('showcase-filename'),
    showcaseScore: $('showcase-score'),
    btnContinue: $('btn-continue'),
  };

  const MAX_ATTEMPTS_BEFORE_PAUSE = 20000;
  const MAX_FILMSTRIP_TILES = 400;
  const MAX_PREVIEW_TILES = 20;
  const SSIM_WINDOW = 3;
  const CELL_SMALL = 6;    // main generation filmstrip
  const CELL_THUMB = 4;    // dataset galleries
  const CELL_BIG = 12;     // match-found comparison
  const CELL_PREVIEW = 5;  // training quality-check previews

  const state = {
    dataset: null,
    vocab: null,
    H: 0, W: 0,

    // Two separate sets, two separate jobs, and they never mix:
    trainTexts: [],  // raw texts, for building the training tensor
    trainSet: [],    // [{ file, text, bin }] — quality gate reads ONLY this
    testSet: [],     // [{ file, text, bin }] — generation matching reads ONLY this
    style: null,

    model: null,
    schedule: null,
    optimizer: null,
    trainTensorAll: null,
    lossHistory: [],
    stepCount: 0,
    qualityMet: false,
    pretrainedActive: false, // true only while state.model IS the fetched pretrained model, not something trained in this tab

    generating: false,
    attempts: 0,
    genStartTime: 0,
    tilesInStrip: 0,
    bestScoreEver: -1,
    bestHammingEver: 0,
    bestFileEver: null,
  };

  async function init() {
    els.buildBadge.textContent = window.BUILD_ID || 'unset';

    try {
      const res = await fetch('data/dataset.json', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.dataset = await res.json();
    } catch (err) {
      els.backend.textContent = 'dataset failed to load';
      els.backend.className = 'readout-value warn';
      els.trainStatus.textContent =
        'Could not load data/dataset.json. Run "node scripts/generate_placeholder_pack.js" then ' +
        '"node scripts/build_dataset.js" (or push — the GitHub Action runs the build step for you) and reload.';
      console.error(err);
      return;
    }

    const d = state.dataset;
    state.H = d.height;
    state.W = d.width;
    state.vocab = d.vocab;

    state.trainTexts = d.train.map((r) => r.text);
    state.trainSet = d.train.map((r) => ({
      file: r.file,
      text: r.text,
      bin: AsciiSimilarity.textToBinary(r.text, state.vocab, state.H, state.W),
    }));
    // The held-out set. Nothing in the training loop is allowed to read
    // this array — only generationLoop()/onMatchFound() below may.
    state.testSet = d.test.map((r) => ({
      file: r.file,
      text: r.text,
      bin: AsciiSimilarity.textToBinary(r.text, state.vocab, state.H, state.W),
    }));

    state.style = AsciiRender.buildCharStyle(state.vocab, state.trainTexts.concat(d.test.map((r) => r.text)));

    els.statTrain.textContent = d.train.length;
    els.statTest.textContent = d.test.length;
    els.statVocab.textContent = state.vocab.length;
    els.statGrid.textContent = `${state.W} × ${state.H}`;

    fillGallery(els.trainGallery, d.train, CELL_THUMB);
    fillGallery(els.testGallery, d.test, CELL_THUMB);

    await tf.ready();
    const backend = tf.getBackend();
    els.backend.textContent = backend;
    els.backend.className = backend === 'webgl' ? 'readout-value ok' : 'readout-value warn';
    if (backend !== 'webgl') {
      els.backendNote.textContent =
        `Running on the "${backend}" backend rather than webgl — training will be much slower. If this is unexpected, try a different browser or check that hardware acceleration is enabled.`;
    }

    els.btnTrain.disabled = false;
    els.trainStatus.textContent = 'Dataset loaded. Set your parameters and train.';

    await loadPretrainedFromServer();
    await loadShowcase();
  }

  // Optional: a result precomputed by `python search.py --showcase-out
  // data/showcase.json` (see README_LOCAL.md), shown immediately without
  // requiring a visitor to train or search anything themselves. Absence of
  // the file is normal, not an error - most visitors won't have generated
  // one, and the rest of the page works identically either way.
  async function loadShowcase() {
    let s;
    try {
      const res = await fetch('data/showcase.json', { cache: 'no-store' });
      if (!res.ok) return;
      s = await res.json();
    } catch (err) {
      return;
    }
    els.showcasePanel.hidden = false;
    AsciiRender.renderGridToCanvas(els.showcaseGenCanvas, s.generatedText,
      { height: state.H, width: state.W, cell: CELL_BIG, style: state.style });
    AsciiRender.renderGridToCanvas(els.showcaseTestCanvas, s.heldOutText,
      { height: state.H, width: state.W, cell: CELL_BIG, style: state.style });
    els.showcaseFilename.textContent = s.heldOutFile;
    const when = s.createdAt ? new Date(s.createdAt).toLocaleDateString() : 'an earlier run';
    els.showcaseMeta.textContent = `model step ${Number(s.modelStep).toLocaleString()} · found ${when}`;
    els.showcaseScore.textContent =
      `${(s.ssim * 100).toFixed(1)}% SSIM · ${(s.hamming * 100).toFixed(1)}% characters identical · ` +
      `found after ${Number(s.attempts).toLocaleString()} attempts (${Number(s.elapsedSeconds).toFixed(0)}s)`;
  }

  function fillGallery(container, entries, cell) {
    container.innerHTML = '';
    for (const { file, text } of entries) {
      const canvas = document.createElement('canvas');
      canvas.title = file;
      AsciiRender.renderGridToCanvas(canvas, text, {
        height: state.H, width: state.W, cell, style: state.style,
      });
      container.appendChild(canvas);
    }
  }

  // ---------------- training ----------------

  function drawLossCurve() {
    const canvas = els.lossCanvas;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.clientWidth * devicePixelRatio;
    const h = canvas.height = canvas.clientHeight * devicePixelRatio;
    ctx.clearRect(0, 0, w, h);
    const hist = state.lossHistory;
    if (hist.length < 2) return;
    const max = Math.max(...hist);
    const min = Math.min(...hist);
    const range = Math.max(max - min, 1e-6);
    ctx.beginPath();
    ctx.strokeStyle = '#6C63FF';
    ctx.lineWidth = 1.5 * devicePixelRatio;
    hist.forEach((v, i) => {
      const x = (i / (hist.length - 1)) * w;
      const y = h - ((v - min) / range) * (h * 0.85) - (h * 0.075);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Draw a handful of fresh samples and score each against its nearest
  // TRAINING image (never the held-out set — see the note in init() and in
  // the "Quality gate" copy on the page). This is the concrete stand-in for
  // "actually good at producing images similar to the training set" — a
  // number the training loop can gate on, instead of a step count picked in
  // advance.
  async function evaluateQuality(n) {
    const samples = [];
    for (let i = 0; i < n; i++) {
      const sampleTensor = DiffusionModel.sampleOne(state.model, state.schedule);
      const text = DiffusionData.analogTensorToText(sampleTensor, state.vocab, state.H, state.W);
      sampleTensor.dispose();
      const bin = AsciiSimilarity.textToBinary(text, state.vocab, state.H, state.W);
      const best = AsciiSimilarity.bestMatch(bin, state.trainSet, state.H, state.W, SSIM_WINDOW);
      samples.push({ text, best });
      await tf.nextFrame();
    }
    const scores = samples.map((s) => s.best.ssim);
    const meanBest = scores.reduce((a, b) => a + b, 0) / scores.length;
    const maxBest = Math.max(...scores);
    return { meanBest, maxBest, samples };
  }

  function renderTrainPreview(samples, targetQuality) {
    els.trainPreview.innerHTML = '';
    for (const { text, best } of samples) {
      const wrap = document.createElement('div');
      wrap.className = 'preview-tile';
      const canvas = document.createElement('canvas');
      canvas.classList.add('tile-in');
      AsciiRender.renderGridToCanvas(canvas, text, {
        height: state.H, width: state.W, cell: CELL_PREVIEW, style: state.style,
      });
      if (best.ssim >= targetQuality) canvas.classList.add('is-close');
      wrap.appendChild(canvas);
      const label = document.createElement('span');
      label.className = 'preview-score';
      label.textContent = `${(best.ssim * 100).toFixed(0)}%`;
      wrap.appendChild(label);
      els.trainPreview.appendChild(wrap);
      while (els.trainPreview.children.length > MAX_PREVIEW_TILES) {
        els.trainPreview.removeChild(els.trainPreview.firstChild);
      }
    }
  }

  async function trainModel() {
    els.btnTrain.disabled = true;
    els.btnTrainMore.hidden = true;
    els.btnGenerate.disabled = true;
    els.matchPanel.hidden = true;
    state.lossHistory = [];
    state.stepCount = 0;
    state.qualityMet = false;
    state.pretrainedActive = false;

    const hidden = parseInt(els.hpHidden.value, 10);
    const T = parseInt(els.hpT.value, 10);
    const lr = parseFloat(els.hpLr.value);

    state.model = DiffusionModel.buildModel({
      height: state.H, width: state.W, vocabSize: state.vocab.length, hidden, timeDim: 32,
    });
    state.model.warmUp();
    state.schedule = DiffusionModel.makeSchedule(T);
    state.optimizer = tf.train.adam(lr);

    if (state.trainTensorAll) state.trainTensorAll.dispose();
    state.trainTensorAll = DiffusionData.batchToTensor(state.trainTexts, state.vocab, state.H, state.W);

    els.trainStatus.textContent =
      `Model ready — ${state.model.countParams().toLocaleString()} parameters. Starting training…`;
    await tf.nextFrame();

    const maxSteps = parseInt(els.hpSteps.value, 10);
    await runTrainingSteps(maxSteps);
  }

  async function runTrainingSteps(maxAdditionalSteps) {
    const batchSize = Math.min(parseInt(els.hpBatch.value, 10), state.trainTexts.length);
    const targetQuality = parseFloat(els.hpTrainTarget.value);
    const evalEvery = parseInt(els.hpEvalEvery.value, 10);
    const evalSamples = parseInt(els.hpEvalSamples.value, 10);

    const t0 = performance.now();
    let stepsThisRun = 0;
    let lastLoss = state.lossHistory[state.lossHistory.length - 1] || 0;

    while (stepsThisRun < maxAdditionalSteps) {
      const idxArr = Array.from({ length: batchSize }, () => Math.floor(Math.random() * state.trainTexts.length));
      const loss = tf.tidy(() => {
        const idxT = tf.tensor1d(idxArr, 'int32');
        const batchX = state.trainTensorAll.gather(idxT);
        return DiffusionModel.trainStep(state.model, state.optimizer, batchX, state.schedule);
      });
      state.lossHistory.push(loss);
      lastLoss = loss;
      state.stepCount++;
      stepsThisRun++;

      const doEval = (state.stepCount % evalEvery === 0);

      if (stepsThisRun % 5 === 0 || doEval || stepsThisRun === maxAdditionalSteps) {
        const elapsed = (performance.now() - t0) / 1000;
        const stepsPerSec = stepsThisRun / Math.max(elapsed, 0.001);
        els.trainProgress.style.width = `${Math.min(100, (stepsThisRun / maxAdditionalSteps) * 100)}%`;
        els.trainStatus.textContent =
          `step ${state.stepCount.toLocaleString()} · loss ${lastLoss.toFixed(4)} · ${stepsPerSec.toFixed(1)} steps/s`;
        drawLossCurve();
        await tf.nextFrame();
      }

      if (doEval) {
        const quality = await evaluateQuality(evalSamples); // trainSet only — see evaluateQuality()
        renderTrainPreview(quality.samples, targetQuality);
        els.qualityStatus.textContent =
          `step ${state.stepCount.toLocaleString()} · mean best-SSIM ${(quality.meanBest * 100).toFixed(1)}% ` +
          `vs. training set (target ${(targetQuality * 100).toFixed(0)}%) · best of batch ${(quality.maxBest * 100).toFixed(1)}%`;
        if (quality.meanBest >= targetQuality) {
          finishTraining(true);
          return;
        }
      }
    }

    finishTraining(false);
  }

  function finishTraining(reachedTarget) {
    els.btnTrain.disabled = false;
    if (reachedTarget) {
      state.qualityMet = true;
      els.btnTrainMore.hidden = true;
      els.trainStatus.textContent += ' · quality target reached, training done';
      els.btnGenerate.disabled = false;
    } else {
      els.trainStatus.textContent += ' · max steps reached without hitting the quality target';
      els.btnTrainMore.hidden = false;
      els.btnTrainMore.disabled = false;
      els.btnGenerate.disabled = !state.qualityMet;
    }
  }

  // Reassembles a bundle written by python/ascii_diffusion/bridge.py's
  // export_model_chunked(): a small manifest.json (meta + the ordered list
  // of chunk files) plus part-000.txt, part-001.txt, ... - plain-text
  // fragments of one big JSON string, split purely by character count with
  // no regard for layer boundaries. Concatenating them back in order and
  // parsing produces exactly the same {weights: {...}} object
  // export_model()'s single file would have held - this exists purely so
  // no single committed file has to hold the whole model, however big it
  // gets. Throws if the manifest itself isn't found (caller decides what
  // that means); manifestUrl's directory is where the parts are expected.
  async function fetchChunkedWeights(manifestUrl) {
    const res = await fetch(manifestUrl, { cache: 'no-store' });
    if (!res.ok) throw new Error(`no manifest at ${manifestUrl}`);
    const manifest = await res.json();
    const base = manifestUrl.slice(0, manifestUrl.lastIndexOf('/') + 1);
    const parts = await Promise.all(
      manifest.chunkFiles.map((name) => fetch(base + name, { cache: 'no-store' }).then((r) => {
        if (!r.ok) throw new Error(`missing chunk "${name}"`);
        return r.text();
      }))
    );
    const payload = parts.join('');
    if (manifest.totalLength != null && payload.length !== manifest.totalLength) {
      throw new Error(`reassembled length ${payload.length} != manifest's ${manifest.totalLength} - a chunk is truncated or missing`);
    }
    const { weights } = JSON.parse(payload);
    return { meta: manifest.meta, weights };
  }

  // ---------------- model source: use existing vs. train your own ----------------

  // Shared by both ways of getting a pretrained model in: the auto-fetched
  // data/weights.json (loadPretrainedFromServer) and the manual file picker
  // (handleLoadWeightsFile). bundle.meta carries the exact buildModel()
  // config the checkpoint was trained with — a model is built to match it
  // exactly, then the weights are handed to DiffusionModel.loadWeights().
  // Nothing here touches state.testSet; this only changes where the
  // model's weights came from, not the matching invariant in
  // generationLoop() below. Returns a human-readable status string;
  // throws on anything invalid (caller decides how to report it).
  function applyWeightsBundle(bundle, sourceLabel) {
    const meta = bundle && bundle.meta;
    if (!meta || !bundle.weights) throw new Error('not a recognized weights.json (missing meta/weights)');
    if (meta.height !== state.H || meta.width !== state.W || meta.vocabSize !== state.vocab.length) {
      throw new Error(
        `this checkpoint is ${meta.width}x${meta.height} grid / ${meta.vocabSize}-symbol vocab, ` +
        `but the loaded dataset is ${state.W}x${state.H} / ${state.vocab.length}-symbol — they must match.`
      );
    }

    els.btnTrain.disabled = true;
    els.btnTrainMore.hidden = true;
    els.btnGenerate.disabled = true;
    els.matchPanel.hidden = true;
    state.lossHistory = [];
    state.stepCount = meta.step || 0;

    const model = DiffusionModel.buildModel({
      height: state.H, width: state.W, vocabSize: state.vocab.length,
      hidden: meta.hidden, timeDim: meta.timeDim,
      bottleneckBlocks: meta.bottleneckBlocks, bottleneckKernel: meta.bottleneckKernel,
    });
    model.warmUp();
    DiffusionModel.loadWeights(model, bundle);

    if (state.trainTensorAll) { state.trainTensorAll.dispose(); state.trainTensorAll = null; }
    state.pretrainedActive = true;
    state.model = model;
    state.schedule = DiffusionModel.makeSchedule(meta.timesteps || parseInt(els.hpT.value, 10));
    state.qualityMet = true;

    const qualityNote = meta.meanQuality != null ? ` · trained mean best-SSIM ${(meta.meanQuality * 100).toFixed(1)}%` : '';
    els.trainStatus.textContent = 'Pretrained model loaded — ready to generate.';
    els.btnGenerate.disabled = false;
    return `Loaded ${sourceLabel}: ${model.countParams().toLocaleString()} params, ` +
      `${meta.bottleneckBlocks}×kernel-${meta.bottleneckKernel} bottleneck, step ${state.stepCount.toLocaleString()}${qualityNote}.`;
  }

  async function handleLoadWeightsFile(e) {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    els.pretrainedStatus.textContent = `Reading ${file.name}…`;
    try {
      const bundle = JSON.parse(await file.text());
      els.pretrainedStatus.textContent = applyWeightsBundle(bundle, file.name);
      setMode('existing');
    } catch (err) {
      console.error(err);
      els.pretrainedStatus.textContent = `Could not load ${file.name}: ${err.message}`;
      els.btnTrain.disabled = false;
    } finally {
      e.target.value = ''; // allow re-selecting the same file later
    }
  }

  // Fetches and applies whatever pretrained model this repo has committed
  // (the chunked data/weights/ directory train.py now produces by default,
  // or a single legacy data/weights.json). Called once automatically on
  // page load, and again any time "Use existing model" is clicked while
  // it isn't already the active model (e.g. after switching to "Train
  // your own" and training something in this tab) - so the button
  // actually does something, not just toggles which panel is visible.
  // Absence of either file is completely normal (most forks won't have
  // trained+committed one yet) and falls back to "Train your own" -
  // never treated as an error.
  async function loadPretrainedFromServer() {
    try {
      const bundle = await fetchChunkedWeights('data/weights/manifest.json');
      els.pretrainedStatus.textContent = applyWeightsBundle(bundle, 'data/weights/ (chunked)');
      setMode('existing');
      return;
    } catch (chunkedErr) {
      // fall through to the legacy single-file path below
    }
    try {
      const res = await fetch('data/weights.json', { cache: 'no-store' });
      if (!res.ok) throw new Error('none found');
      const bundle = await res.json();
      els.pretrainedStatus.textContent = applyWeightsBundle(bundle, 'data/weights.json');
      setMode('existing');
    } catch (err) {
      els.pretrainedStatus.textContent =
        'No pretrained model found (checked data/weights/manifest.json and data/weights.json) — ' +
        'load one below, or switch to "Train your own".';
      setMode('train');
    }
  }

  // Toggles which half of the Model panel is visible. Both halves stay
  // fully built/wired regardless of mode — this only changes what's
  // shown, not what state exists, so switching back and forth (e.g. to
  // try training after loading a pretrained model) always works.
  function setMode(mode) {
    const existing = mode === 'existing';
    els.modeExistingContent.hidden = !existing;
    els.modeTrainContent.hidden = existing;
    els.btnModeExisting.classList.toggle('primary', existing);
    els.btnModeTrain.classList.toggle('primary', !existing);
    // applyWeightsBundle() disables btnTrain while a pretrained model is
    // active (no reason to show it clickable in that mode) - switching
    // back to "Train your own" needs to undo that, or the button would
    // stay stuck disabled with no way to reach it again. Safe to just
    // re-enable unconditionally here: reaching this code at all means the
    // dataset already loaded successfully, which is the only other
    // precondition trainModel() needs.
    if (!existing) els.btnTrain.disabled = false;
  }

  // ---------------- generation ----------------


  function addFilmstripTile(text, tag) {
    const canvas = document.createElement('canvas');
    canvas.classList.add('tile-in');
    AsciiRender.renderGridToCanvas(canvas, text, {
      height: state.H, width: state.W, cell: CELL_SMALL, style: state.style,
    });
    if (tag === 'match') canvas.classList.add('is-match');
    else if (tag === 'best') canvas.classList.add('is-close');
    els.filmstrip.appendChild(canvas);
    state.tilesInStrip++;
    while (state.tilesInStrip > MAX_FILMSTRIP_TILES) {
      els.filmstrip.removeChild(els.filmstrip.firstChild);
      state.tilesInStrip--;
    }
    els.filmstrip.scrollLeft = els.filmstrip.scrollWidth;
  }

  function updateGenStatus() {
    const elapsed = (performance.now() - state.genStartTime) / 1000;
    const rate = state.attempts / Math.max(elapsed, 0.001);
    els.genStatus.textContent =
      `attempt ${state.attempts.toLocaleString()} · ${rate.toFixed(1)}/s · ${elapsed.toFixed(0)}s elapsed`;
  }

  function updateBestStatus(threshold) {
    if (state.bestScoreEver < 0) { els.bestStatus.textContent = 'No attempts yet.'; return; }
    els.bestStatus.textContent =
      `best so far: ${(state.bestScoreEver * 100).toFixed(1)}% SSIM vs held-out file ${state.bestFileEver} ` +
      `(${(state.bestHammingEver * 100).toFixed(1)}% characters identical) · target ${(threshold * 100).toFixed(0)}%`;
  }

  // Sampling a wave at once amortizes the T reverse-diffusion model calls
  // across WAVE_SIZE grids instead of paying them per single sample, and
  // lets us yield to the browser once per wave instead of once per attempt
  // (the actual per-attempt bottleneck: one <canvas> + one tf.nextFrame()
  // yield, each of which costs far more than the sampling itself for a
  // model this small). Scoring stays exactly as before — one pair at a time
  // via similarity.js, still against testSet only. For a held-out set in
  // the tens of thousands, python/search.py's vectorized scoring is the
  // right tool for real throughput; this just removes the easy overhead
  // from the in-browser demo.
  const WAVE_SIZE = 8;

  // Every generated sample is scored only against state.testSet — the
  // held-out files the model never trained on. This is the whole
  // demonstration; scoring against state.trainSet here would be meaningless.
  async function generationLoop() {
    const threshold = parseFloat(els.hpThreshold.value);
    while (state.generating) {
      const waveTensor = DiffusionModel.sampleBatch(state.model, state.schedule, WAVE_SIZE);
      for (let i = 0; i < WAVE_SIZE; i++) {
        if (!state.generating) { waveTensor.dispose(); return; }

        const one = waveTensor.slice([i, 0, 0, 0], [1, state.H, state.W, state.vocab.length]);
        const text = DiffusionData.analogTensorToText(one, state.vocab, state.H, state.W);
        one.dispose();
        state.attempts++;

        const bin = AsciiSimilarity.textToBinary(text, state.vocab, state.H, state.W);
        const best = AsciiSimilarity.bestMatch(bin, state.testSet, state.H, state.W, SSIM_WINDOW);

        let tag = null;
        if (best.ssim > state.bestScoreEver) {
          state.bestScoreEver = best.ssim;
          state.bestHammingEver = best.hamming;
          state.bestFileEver = best.file;
          tag = 'best';
        }
        const isMatch = best.ssim >= threshold;
        if (isMatch) tag = 'match';

        addFilmstripTile(text, tag);

        if (isMatch) {
          waveTensor.dispose();
          updateGenStatus();
          updateBestStatus(threshold);
          onMatchFound(text, best);
          return;
        }
        if (state.attempts >= MAX_ATTEMPTS_BEFORE_PAUSE) {
          waveTensor.dispose();
          state.generating = false;
          updateGenStatus();
          updateBestStatus(threshold);
          els.genStatus.textContent += ' · paused — no match yet, press start to keep going';
          els.btnGenerate.disabled = false;
          els.btnStop.disabled = true;
          return;
        }
      }
      waveTensor.dispose();
      updateGenStatus();
      updateBestStatus(threshold);
      await tf.nextFrame();
    }
  }

  function onMatchFound(text, best) {
    state.generating = false;
    els.btnGenerate.disabled = true;
    els.btnStop.disabled = true;

    const elapsed = ((performance.now() - state.genStartTime) / 1000).toFixed(1);
    els.matchMeta.textContent = `found on attempt ${state.attempts.toLocaleString()}, after ${elapsed}s`;

    AsciiRender.renderGridToCanvas(els.matchGenCanvas, text, {
      height: state.H, width: state.W, cell: CELL_BIG, style: state.style,
    });
    AsciiRender.renderGridToCanvas(els.matchTestCanvas, best.text, {
      height: state.H, width: state.W, cell: CELL_BIG, style: state.style,
    });
    els.matchFilename.textContent = best.file;

    const diff = AsciiRender.diffCount(text, best.text);
    els.matchScore.textContent =
      `${(best.ssim * 100).toFixed(1)}% SSIM · ${(best.hamming * 100).toFixed(1)}% characters identical ` +
      `(${diff} of ${state.H * state.W} differ).`;

    els.btnDownloadMatch.onclick = () => downloadText(text, best.file);
    els.matchPanel.hidden = false;
    els.matchPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function downloadText(text, filename) {
    const blob = new Blob([text + '\n'], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `generated_${filename}`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function startGenerating() {
    if (!state.model) return;
    state.generating = true;
    state.attempts = 0;
    state.bestScoreEver = -1;
    state.bestHammingEver = 0;
    state.bestFileEver = null;
    state.genStartTime = performance.now();
    els.btnGenerate.disabled = true;
    els.btnStop.disabled = false;
    els.matchPanel.hidden = true;
    els.bestStatus.textContent = 'No attempts yet.';
    generationLoop();
  }

  // ---------------- wire up ----------------

  els.btnTrain.addEventListener('click', trainModel);
  els.btnTrainMore.addEventListener('click', async () => {
    els.btnTrainMore.disabled = true;
    els.btnTrain.disabled = true;
    const more = parseInt(els.hpSteps.value, 10);
    await runTrainingSteps(more);
  });
  els.loadWeightsFile.addEventListener('change', handleLoadWeightsFile);
  els.btnModeExisting.addEventListener('click', async () => {
    if (state.pretrainedActive) {
      setMode('existing'); // already loaded and active - just show it, no need to refetch
    } else {
      await loadPretrainedFromServer(); // e.g. switching back after training something in this tab
    }
  });
  els.btnModeTrain.addEventListener('click', () => setMode('train'));
  els.btnGenerate.addEventListener('click', startGenerating);
  els.btnStop.addEventListener('click', () => {
    state.generating = false;
    els.btnStop.disabled = true;
    els.btnGenerate.disabled = false;
  });
  els.btnContinue.addEventListener('click', () => {
    els.matchPanel.hidden = true;
    startGenerating();
  });

  init();
})();
