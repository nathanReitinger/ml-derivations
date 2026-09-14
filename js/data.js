(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory(require('@tensorflow/tfjs'));
  } else {
    root.DiffusionData = factory(root.tf);
  }
})(typeof self !== 'undefined' ? self : this, function (tf) {

  function normalizeText(text) {
    return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').replace(/\n+$/, '');
  }

  function textToRows(text, height, width) {
    const rows = normalizeText(text).split('\n');
    if (rows.length !== height) {
      throw new Error(`expected ${height} rows, got ${rows.length}`);
    }
    for (const r of rows) {
      if (r.length !== width) throw new Error(`expected ${width} cols, got ${r.length} in row "${r}"`);
    }
    return rows;
  }

  function buildVocab(texts, height, width) {
    const set = new Set();
    for (const t of texts) {
      for (const ch of normalizeText(t)) {
        if (ch !== '\n') set.add(ch);
      }
    }
    return Array.from(set).sort();
  }

  function textToIndices(text, vocab, height, width) {
    const rows = textToRows(text, height, width);
    const idx = new Int32Array(height * width);
    const map = new Map(vocab.map((c, i) => [c, i]));
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const ch = rows[r][c];
        const v = map.get(ch);
        if (v === undefined) throw new Error(`character "${ch}" not in vocab`);
        idx[r * width + c] = v;
      }
    }
    return idx;
  }

  function indicesToText(idx, vocab, height, width) {
    const rows = [];
    for (let r = 0; r < height; r++) {
      let row = '';
      for (let c = 0; c < width; c++) row += vocab[idx[r * width + c]];
      rows.push(row);
    }
    return rows.join('\n');
  }

  // signed one-hot ("analog bits"): shape [height, width, vocabSize], values in {-1, +1}
  function textToAnalogTensor(text, vocab, height, width) {
    const idx = textToIndices(text, vocab, height, width);
    return tf.tidy(() => {
      const idxT = tf.tensor1d(Array.from(idx), 'int32').reshape([height, width]);
      const oneHot = tf.oneHot(idxT, vocab.length).toFloat();
      return oneHot.mul(2).sub(1);
    });
  }

  function batchToTensor(texts, vocab, height, width) {
    const tensors = texts.map(t => textToAnalogTensor(t, vocab, height, width));
    const batched = tf.stack(tensors);
    tensors.forEach(t => t.dispose());
    return batched;
  }

  // decode a [1,H,W,V] (or [H,W,V]) tensor of continuous values back to text via per-cell argmax
  function analogTensorToText(tensor, vocab, height, width) {
    return tf.tidy(() => {
      const t = tensor.shape.length === 4 ? tensor.squeeze([0]) : tensor;
      const idx = t.argMax(-1);
      const arr = idx.dataSync();
      return indicesToText(arr, vocab, height, width);
    });
  }

  return {
    normalizeText, textToRows, buildVocab, textToIndices, indicesToText,
    textToAnalogTensor, batchToTensor, analogTensorToText,
  };
});
