const tf = require('@tensorflow/tfjs');
const DiffusionModel = require('../js/model.js');
const DiffusionData = require('../js/data.js');

const H = 10, W = 20, VOCAB = 2;
const vocab = ['.', '#'];
const model = DiffusionModel.buildModel({
  height: H, width: W, vocabSize: VOCAB, hidden: 8, timeDim: 16, bottleneckBlocks: 2, bottleneckKernel: 3,
});
model.warmUp();
const schedule = DiffusionModel.makeSchedule(5); // tiny T, just exercising the plumbing

const WAVE = 6;
const waveTensor = DiffusionModel.sampleBatch(model, schedule, WAVE);
console.log('waveTensor shape:', waveTensor.shape, '(expect', [WAVE, H, W, VOCAB], ')');

// Reference: decode the whole batch at once via argmax directly.
const wholeBatchIdx = waveTensor.argMax(-1).arraySync(); // [WAVE, H, W]

let ok = true;
for (let i = 0; i < WAVE; i++) {
  const one = waveTensor.slice([i, 0, 0, 0], [1, H, W, VOCAB]);
  const text = DiffusionData.analogTensorToText(one, vocab, H, W);
  one.dispose();

  // Re-derive indices from the decoded text and compare to the whole-batch argmax for row i.
  const rows = text.split('\n');
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const decodedIdx = vocab.indexOf(rows[r][c]);
      const refIdx = wholeBatchIdx[i][r][c];
      if (decodedIdx !== refIdx) {
        ok = false;
        console.log(`MISMATCH sample ${i} at (${r},${c}): decoded=${decodedIdx} ref=${refIdx}`);
      }
    }
  }
}
waveTensor.dispose();
console.log(ok ? 'PASS - slice+decode matches whole-batch argmax for every sample' : 'FAIL');
