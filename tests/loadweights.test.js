const tf = require('@tensorflow/tfjs');
const DiffusionModel = require('../js/model.js');

const H = 10, W = 20, VOCAB = 2, HIDDEN = 12, TIMEDIM = 24, BLOCKS = 4, KERNEL = 5;
const cfg = { height: H, width: W, vocabSize: VOCAB, hidden: HIDDEN, timeDim: TIMEDIM,
               bottleneckBlocks: BLOCKS, bottleneckKernel: KERNEL };

const modelA = DiffusionModel.buildModel(cfg);
modelA.warmUp();

function dumpLayer(layer) {
  const out = {};
  for (const w of layer.getWeights()) {
    const key = w.name.includes('kernel') ? 'kernel' : w.name.includes('bias') ? 'bias'
              : w.name.includes('gamma') ? 'gamma' : w.name.includes('beta') ? 'beta' : w.name;
    out[key] = { shape: w.shape, data: Array.from(w.dataSync()) };
  }
  return out;
}
function dumpModel(model) {
  const w = {};
  for (const [name, layer] of Object.entries(model.layers)) {
    if (name === 'bottleneck') {
      w.bottleneck = layer.map(block => {
        const b = {};
        for (const [sub, l] of Object.entries(block)) b[sub] = dumpLayer(l);
        return b;
      });
    } else if (layer && layer.getWeights) {
      w[name] = dumpLayer(layer);
    }
  }
  return { meta: { height: model.height, width: model.width, vocabSize: model.vocabSize,
                    hidden: HIDDEN, timeDim: model.timeDim,
                    bottleneckBlocks: model.bottleneckBlocks, bottleneckKernel: model.bottleneckKernel },
           weights: w };
}

const bundle = dumpModel(modelA);

// Model B: freshly built (different random init), then overwritten via loadWeights.
const modelB = DiffusionModel.buildModel(cfg);
modelB.warmUp();
DiffusionModel.loadWeights(modelB, bundle);

// Compare on a fresh random input/self-cond/timestep.
const B = 3;
const x = tf.randomNormal([B, H, W, VOCAB]);
const sc = tf.randomNormal([B, H, W, VOCAB]);
const temb = DiffusionModel.timeEmbeddingBatch(Float32Array.from([0.1, 0.5, 0.9]), TIMEDIM);

const outA = modelA.forward(x, temb, sc);
const outB = modelB.forward(x, temb, sc);
const diff = outA.sub(outB).abs().max().dataSync()[0];
console.log('bottleneckBlocks resolved on modelA:', modelA.bottleneckBlocks, 'kernel:', modelA.bottleneckKernel);
console.log('max abs diff between A (source) and B (loaded via loadWeights):', diff);
console.log(diff === 0 ? 'PASS - exact match' : 'FAIL');

// Also sanity-check sampleBatch runs and returns the right shape.
const schedule = DiffusionModel.makeSchedule(6); // tiny T just to smoke-test shape/plumbing, not quality
const samples = DiffusionModel.sampleBatch(modelA, schedule, 4);
console.log('sampleBatch output shape:', samples.shape, '(expect [4,10,20,2])');
