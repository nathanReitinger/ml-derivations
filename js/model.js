(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('@tensorflow/tfjs'));
  } else {
    root.DiffusionModel = factory(root.tf);
  }
})(typeof self !== 'undefined' ? self : this, function (tf) {

  // ---- noise schedule ----------------------------------------------------
  // Cosine schedule (Nichol & Dhariwal, "Improved DDPM"). At low step counts
  // (T ~ 50-100, which is what we can afford for interactive in-browser
  // sampling) this keeps far more of the signal-to-noise curve in a useful
  // range than a linear beta schedule, which spends too many steps at
  // "almost pure noise" and starves the model of gradient signal where it
  // matters. i=0 is least noised, i=T-1 is most noised (matches the rest of
  // the codebase's convention).
  function makeSchedule(T) {
    const s = 0.008;
    const f = (t) => Math.cos(((t / T + s) / (1 + s)) * Math.PI / 2) ** 2;
    const f0 = f(0);
    const betas = [], alphas = [], alphaBars = [];
    let prevAlphaBar = 1;
    for (let i = 0; i < T; i++) {
      const abar = f(i + 1) / f0;
      const beta = Math.min(Math.max(1 - abar / prevAlphaBar, 1e-5), 0.999);
      betas.push(beta);
      alphas.push(1 - beta);
      alphaBars.push(abar);
      prevAlphaBar = abar;
    }
    return { T, betas, alphas, alphaBars };
  }

  function timeEmbeddingBatch(tNormArr, dim) {
    return tf.tidy(() => {
      const half = Math.max(Math.floor(dim / 2), 1);
      const freqs = [];
      for (let i = 0; i < half; i++) freqs.push(Math.exp(-Math.log(10000) * i / Math.max(half - 1, 1)));
      const freqTensor = tf.tensor2d(freqs, [1, half]);
      const t = tf.tensor2d(Array.from(tNormArr), [tNormArr.length, 1]);
      const args = t.matMul(freqTensor).mul(2 * Math.PI);
      let emb = tf.concat([tf.sin(args), tf.cos(args)], 1);
      if (emb.shape[1] < dim) {
        emb = tf.pad(emb, [[0, 0], [0, dim - emb.shape[1]]]);
      } else if (emb.shape[1] > dim) {
        emb = emb.slice([0, 0], [emb.shape[0], dim]);
      }
      return emb;
    });
  }

  const BOTTLENECK_LAYERS = 6;
  const BOTTLENECK_KERNEL = 7;
  const SELF_COND_PROB = 0.5;
  const X0_CLIP = 1.5;

  function buildModel({
    height, width, vocabSize, hidden = 48, timeDim = 32,
    bottleneckBlocks = BOTTLENECK_LAYERS, bottleneckKernel = BOTTLENECK_KERNEL,
  }) {
    if (height % 2 !== 0 || width % 2 !== 0) {
      throw new Error(
        `buildModel: height and width must both be even (got ${width}x${height}) — ` +
        `the stride-2 downsample + nearest-2x upsample only round-trips back to the exact ` +
        `original size when both are even; an odd dimension would silently produce a ` +
        `shape-mismatch crash deep in forward() instead of this clear one.`
      );
    }

    const makeNorm = () => tf.layers.layerNormalization({ axis: -1 });
    // FiLM projection: temb -> per-channel (scale, shift). Every stage gets
    // its own projection so early/late layers can react to the timestep
    // differently, rather than the single "concat time features once at the
    // input" trick the old model used (which starved everything past the
    // first conv of any explicit timestep signal).
    const makeFilm = (channels) => tf.layers.dense({ units: channels * 2, activation: 'linear' });
    const makeConv = (filters, kernelSize, extra) => tf.layers.conv2d(Object.assign(
      { filters, kernelSize, padding: 'same', activation: 'linear' }, extra || {}));

    const layers = {
      tDense1: tf.layers.dense({ units: 128, activation: 'relu', inputShape: [timeDim] }),
      tDense2: tf.layers.dense({ units: hidden, activation: 'relu' }),

      // input is [x ; self-cond guess] concatenated on channels -> 2*vocabSize
      stemConv: makeConv(hidden, 3),
      stemNorm: makeNorm(),
      stemFilm: makeFilm(hidden),

      downConv: makeConv(hidden, 4, { strides: 2 }),
      downNorm: makeNorm(),
      downFilm: makeFilm(hidden),

      bottleneck: Array.from({ length: bottleneckBlocks }, () => ({
        conv1: makeConv(hidden, bottleneckKernel),
        norm1: makeNorm(),
        film1: makeFilm(hidden),
        conv2: makeConv(hidden, bottleneckKernel),
        norm2: makeNorm(),
        film2: makeFilm(hidden),
      })),

      upSample: tf.layers.upSampling2d({ size: [2, 2] }),

      refine1Conv: makeConv(hidden, 3),
      refine1Norm: makeNorm(),
      refine1Film: makeFilm(hidden),

      refine2Conv: makeConv(hidden, 3),
      refine2Norm: makeNorm(),
      refine2Film: makeFilm(hidden),

      convOut: makeConv(vocabSize, 3),
    };

    function film(h, temb, filmLayer, channels) {
      let f = filmLayer.apply(temb);          // [B, 2*channels]
      f = f.reshape([-1, 1, 1, channels * 2]);
      const [scale, shift] = tf.split(f, 2, -1);
      return h.mul(scale.add(1)).add(shift);
    }

    // conv -> norm -> FiLM(t) -> activation, applied functionally so every
    // stage (not just the input) is conditioned on the timestep.
    function convBlock(h, conv, norm, filmLayer, temb, channels, activate) {
      h = conv.apply(h);
      h = norm.apply(h);
      h = film(h, temb, filmLayer, channels);
      return activate ? tf.relu(h) : h;
    }

    function forward(xBatch, tEmbBatch, selfCondBatch) {
      let temb = layers.tDense1.apply(tEmbBatch);
      temb = layers.tDense2.apply(temb); // [B, hidden]

      const xin = tf.concat([xBatch, selfCondBatch], -1); // [B,H,W,2*vocabSize]

      // Full-resolution stem, kept as a skip connection so fine edges don't
      // have to survive the downsample/upsample round trip.
      let skip = convBlock(xin, layers.stemConv, layers.stemNorm, layers.stemFilm, temb, hidden, true);

      // Downsample 2x. A kernel-7 conv at this resolution has a large
      // effective receptive field back at full resolution, which is what
      // lets the bottleneck reason about global shape rather than only
      // local texture.
      let h = convBlock(skip, layers.downConv, layers.downNorm, layers.downFilm, temb, hidden, true);

      for (const block of layers.bottleneck) {
        let b = convBlock(h, block.conv1, block.norm1, block.film1, temb, hidden, true);
        b = convBlock(b, block.conv2, block.norm2, block.film2, temb, hidden, false);
        h = tf.relu(tf.add(h, b));
      }

      h = layers.upSample.apply(h);
      h = tf.concat([h, skip], -1);

      h = convBlock(h, layers.refine1Conv, layers.refine1Norm, layers.refine1Film, temb, hidden, true);
      let r2 = convBlock(h, layers.refine2Conv, layers.refine2Norm, layers.refine2Film, temb, hidden, false);
      h = tf.relu(tf.add(h, r2));

      return layers.convOut.apply(h);
    }

    function countParams() {
      let n = 0;
      const flat = [];
      (function collect(v) {
        if (Array.isArray(v)) { v.forEach(collect); return; }
        if (v && typeof v === 'object') {
          if (typeof v.getWeights === 'function') { flat.push(v); return; }
          for (const k in v) collect(v[k]);
        }
      })(layers);
      for (const layer of flat) {
        for (const w of layer.getWeights()) n += w.size;
      }
      return n;
    }

    function warmUp() {
      tf.tidy(() => {
        const dummyX = tf.zeros([1, height, width, vocabSize]);
        const dummySC = tf.zeros([1, height, width, vocabSize]);
        const dummyT = tf.zeros([1, timeDim]);
        forward(dummyX, dummyT, dummySC);
      });
    }

    return {
      layers, forward, warmUp, countParams,
      height, width, vocabSize, timeDim, bottleneckBlocks, bottleneckKernel,
    };
  }

  function trainStep(model, optimizer, batchX0, schedule) {
    const B = batchX0.shape[0];
    const T = schedule.T;
    const tIdx = Array.from({ length: B }, () => Math.floor(Math.random() * T));
    const tNorm = Float32Array.from(tIdx.map(i => i / Math.max(T - 1, 1)));
    const alphaBarArr = tIdx.map(i => schedule.alphaBars[i]);

    const { xt, noise, temb } = tf.tidy(() => {
      const noiseT = tf.randomNormal(batchX0.shape);
      const alphaBarT = tf.tensor(alphaBarArr, [B, 1, 1, 1]);
      const sqrtAB = alphaBarT.sqrt();
      const sqrtOMAB = alphaBarT.mul(-1).add(1).sqrt();
      const xtT = batchX0.mul(sqrtAB).add(noiseT.mul(sqrtOMAB));
      const tembT = timeEmbeddingBatch(tNorm, model.timeDim);
      return { xt: xtT, noise: noiseT, temb: tembT };
    });

    // Self-conditioning (Chen, Zhang & Hinton, "Analog Bits"): half the
    // time, bootstrap from the model's own (stop-gradient) estimate of x0
    // instead of always feeding zeros. This is what actually makes the
    // analog-bits encoding this codebase already uses produce sharp,
    // globally-consistent samples instead of a noisy average of everything
    // it's ever seen -- it was the one piece of that recipe missing here.
    let selfCond;
    if (Math.random() < SELF_COND_PROB) {
      selfCond = tf.tidy(() => {
        const zerosSC = tf.zerosLike(batchX0);
        const predNoise0 = model.forward(xt, temb, zerosSC);
        const alphaBarT = tf.tensor(alphaBarArr, [B, 1, 1, 1]);
        const sqrtAB = alphaBarT.sqrt();
        const sqrtOMAB = alphaBarT.mul(-1).add(1).sqrt();
        const x0hat = xt.sub(predNoise0.mul(sqrtOMAB)).div(sqrtAB);
        return x0hat.clipByValue(-X0_CLIP, X0_CLIP);
      });
    } else {
      selfCond = tf.zerosLike(batchX0);
    }

    const lossTensor = optimizer.minimize(() => {
      return tf.tidy(() => {
        const predNoise = model.forward(xt, temb, selfCond);
        return tf.losses.meanSquaredError(noise, predNoise);
      });
    }, true);

    const lossVal = lossTensor.dataSync()[0];
    lossTensor.dispose();
    xt.dispose(); noise.dispose(); temb.dispose(); selfCond.dispose();
    return lossVal;
  }

  function sampleOne(model, schedule, onStep) {
    const T = schedule.T;
    const { height, width, vocabSize, timeDim } = model;
    let x = tf.randomNormal([1, height, width, vocabSize]);
    let selfCond = tf.zeros(x.shape);
    for (let i = T - 1; i >= 0; i--) {
      const tNorm = Float32Array.from([i / Math.max(T - 1, 1)]);
      const alpha = schedule.alphas[i];
      const alphaBar = schedule.alphaBars[i];
      const beta = schedule.betas[i];
      const { xNext, x0hat } = tf.tidy(() => {
        const temb = timeEmbeddingBatch(tNorm, timeDim);
        const predNoise = model.forward(x, temb, selfCond);

        const sqrtAB = Math.sqrt(alphaBar);
        const sqrtOMAB = Math.sqrt(Math.max(1 - alphaBar, 1e-8));
        const x0hatT = x.sub(predNoise.mul(sqrtOMAB)).div(sqrtAB).clipByValue(-X0_CLIP, X0_CLIP);

        const coef1 = 1 / Math.sqrt(alpha);
        const coef2 = beta / sqrtOMAB;
        let mean = x.sub(predNoise.mul(coef2)).mul(coef1);
        if (i > 0) {
          const noise = tf.randomNormal(x.shape);
          mean = mean.add(noise.mul(Math.sqrt(beta)));
        }
        return { xNext: mean, x0hat: x0hatT };
      });
      x.dispose();
      selfCond.dispose();
      x = xNext;
      selfCond = x0hat;
      if (onStep) onStep(i, T);
    }
    selfCond.dispose();
    return x;
  }

  // Batched version of sampleOne: `batchSize` independent reverse-diffusion
  // trajectories, sharing the T model calls (one call scores the whole
  // batch). sampleOne is left untouched above; this is purely additive so a
  // batch of 1 plus this function is a drop-in replacement anywhere sampleOne
  // was used, without touching existing call sites.
  function sampleBatch(model, schedule, batchSize, onStep) {
    const T = schedule.T;
    const { height, width, vocabSize, timeDim } = model;
    let x = tf.randomNormal([batchSize, height, width, vocabSize]);
    let selfCond = tf.zeros(x.shape);
    for (let i = T - 1; i >= 0; i--) {
      const tNorm = new Float32Array(batchSize).fill(i / Math.max(T - 1, 1));
      const alpha = schedule.alphas[i];
      const alphaBar = schedule.alphaBars[i];
      const beta = schedule.betas[i];
      const { xNext, x0hat } = tf.tidy(() => {
        const temb = timeEmbeddingBatch(tNorm, timeDim);
        const predNoise = model.forward(x, temb, selfCond);

        const sqrtAB = Math.sqrt(alphaBar);
        const sqrtOMAB = Math.sqrt(Math.max(1 - alphaBar, 1e-8));
        const x0hatT = x.sub(predNoise.mul(sqrtOMAB)).div(sqrtAB).clipByValue(-X0_CLIP, X0_CLIP);

        const coef1 = 1 / Math.sqrt(alpha);
        const coef2 = beta / sqrtOMAB;
        let mean = x.sub(predNoise.mul(coef2)).mul(coef1);
        if (i > 0) {
          const noise = tf.randomNormal(x.shape);
          mean = mean.add(noise.mul(Math.sqrt(beta)));
        }
        return { xNext: mean, x0hat: x0hatT };
      });
      x.dispose();
      selfCond.dispose();
      x = xNext;
      selfCond = x0hat;
      if (onStep) onStep(i, T);
    }
    selfCond.dispose();
    return x; // [batchSize, height, width, vocabSize]
  }

  // Converts one IEEE-754 half-precision (float16) bit pattern to a JS
  // number. Verified bit-exact against numpy's own float16<->float32
  // conversion on ~25,000 realistic weight values plus edge cases (0, ±1,
  // subnormals, ±65500, inf, nan) before this was written - see
  // python/bridge_test/. This, not a browser-native Float16Array (not
  // reliably available across browsers yet), is what decodes bridge.py's
  // exported weights.
  function float16BitsToNumber(h) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exponent = (h & 0x7C00) >> 10;
    const fraction = h & 0x03FF;
    if (exponent === 0) return sign * fraction * Math.pow(2, -24);
    if (exponent === 0x1F) return fraction ? NaN : sign * Infinity;
    return sign * (1 + fraction / 1024) * Math.pow(2, exponent - 15);
  }

  // One weight value as exported by python/ascii_diffusion/bridge.py is
  // {shape, dtype:'float16', data_b64} - base64 of little-endian float16
  // bytes, chosen there specifically because a JSON array of float32 text
  // runs ~13 bytes/value (185MB for this model's 14M params) versus
  // ~2.1 bytes/value packed this way, which is what keeps the exported file
  // small enough to commit to a repo without Git LFS (which GitHub Pages
  // can't serve from anyway). A plain {shape, data} array (e.g. built
  // directly in JS, as the tests in tests/ do) is also accepted as-is.
  function decodeWeightValue(v) {
    if (v.data_b64 === undefined) return v.data;
    if (v.dtype !== 'float16') throw new Error(`loadWeights: unsupported weight dtype "${v.dtype}"`);
    const binary = atob(v.data_b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const view = new DataView(bytes.buffer);
    const n = bytes.length / 2;
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = float16BitsToNumber(view.getUint16(i * 2, true));
    return out;
  }

  // Apply one exported layer's {kernel,bias} or {gamma,beta} onto a live
  // tf.js layer. Matched by substring on the layer's OWN weight names
  // (rather than assumed positional order) so this doesn't depend on
  // tf.js's internal weight-creation order for any given layer type.
  function setWeightsByName(layer, valuesByRole) {
    const current = layer.getWeights();
    const tensors = current.map((w) => {
      const role = w.name.includes('kernel') ? 'kernel'
        : w.name.includes('bias') ? 'bias'
        : w.name.includes('gamma') ? 'gamma'
        : w.name.includes('beta') ? 'beta' : null;
      if (!role || !valuesByRole || !valuesByRole[role]) {
        throw new Error(`loadWeights: no exported value for weight "${w.name}" (role: ${role})`);
      }
      const v = valuesByRole[role];
      return tf.tensor(decodeWeightValue(v), v.shape);
    });
    layer.setWeights(tensors);
    tensors.forEach((t) => t.dispose());
  }

  // Loads a bundle produced by python/ascii_diffusion/bridge.py's
  // export_model() onto a tf.js model built with a MATCHING config (same
  // height/width/vocabSize/hidden/timeDim/bottleneckBlocks/bottleneckKernel
  // as bundle.meta — build the model with those exact values first).
  function loadWeights(model, bundle) {
    const w = bundle.weights;
    setWeightsByName(model.layers.tDense1, w.tDense1);
    setWeightsByName(model.layers.tDense2, w.tDense2);
    setWeightsByName(model.layers.stemConv, w.stemConv);
    setWeightsByName(model.layers.stemNorm, w.stemNorm);
    setWeightsByName(model.layers.stemFilm, w.stemFilm);
    setWeightsByName(model.layers.downConv, w.downConv);
    setWeightsByName(model.layers.downNorm, w.downNorm);
    setWeightsByName(model.layers.downFilm, w.downFilm);
    if (model.layers.bottleneck.length !== w.bottleneck.length) {
      throw new Error(
        `loadWeights: model has ${model.layers.bottleneck.length} bottleneck blocks, ` +
        `checkpoint has ${w.bottleneck.length} — build the model with bottleneckBlocks: ${w.bottleneck.length}`
      );
    }
    model.layers.bottleneck.forEach((block, i) => {
      const bw = w.bottleneck[i];
      setWeightsByName(block.conv1, bw.conv1);
      setWeightsByName(block.norm1, bw.norm1);
      setWeightsByName(block.film1, bw.film1);
      setWeightsByName(block.conv2, bw.conv2);
      setWeightsByName(block.norm2, bw.norm2);
      setWeightsByName(block.film2, bw.film2);
    });
    setWeightsByName(model.layers.refine1Conv, w.refine1Conv);
    setWeightsByName(model.layers.refine1Norm, w.refine1Norm);
    setWeightsByName(model.layers.refine1Film, w.refine1Film);
    setWeightsByName(model.layers.refine2Conv, w.refine2Conv);
    setWeightsByName(model.layers.refine2Norm, w.refine2Norm);
    setWeightsByName(model.layers.refine2Film, w.refine2Film);
    setWeightsByName(model.layers.convOut, w.convOut);
  }

  return {
    makeSchedule, timeEmbeddingBatch, buildModel, trainStep, sampleOne, sampleBatch, loadWeights,
  };
});
