(function (root) {
  // This whole module exists because the site's goal changed from "prove an
  // exact character-for-character match against data the model never saw"
  // to "keep sampling until one generation closely resembles a training
  // image" — so "closely resembles" needs a real number, not just ===.
  //
  // The grid is tiny (20x10, 2 symbols) so everything here is done in plain
  // JS on typed arrays — no need for tf.js, and it's fast enough to score a
  // generated sample against 50 training images many times a second.

  // Map each character grid to a flat 0/1 array. vocab[0] -> 0, everything
  // else -> 1. Built this way (rather than via render.js's "most frequent
  // character is background" heuristic, which is only for coloring) so it's
  // a pure, deterministic function of the vocab list alone.
  function textToBinary(text, vocab, height, width) {
    const rows = text.split('\n');
    const out = new Uint8Array(height * width);
    const zero = vocab[0];
    for (let r = 0; r < height; r++) {
      const row = rows[r] || '';
      for (let c = 0; c < width; c++) {
        out[r * width + c] = row[c] === zero ? 0 : 1;
      }
    }
    return out;
  }

  // Fraction of cells that are identical (1.0 = exact match).
  function hammingSimilarity(a, b) {
    let same = 0;
    for (let i = 0; i < a.length; i++) if (a[i] === b[i]) same++;
    return same / a.length;
  }

  // Mean structural similarity (Wang et al., 2004) over sliding windows,
  // computed directly on the 0/1 grids (treated as an image with dynamic
  // range L=1). Box-filtered (uniform) windows rather than Gaussian —
  // overkill for a 20x10 binary grid, this is plenty. Only counts windows
  // that fit fully inside the grid (valid, not same-padded).
  //
  // Returns a value clamped to [0, 1]. True SSIM can dip slightly negative
  // for anti-correlated patches; for this UI a single "closeness" percentage
  // is more useful than a signed one, so it's floored at 0.
  function ssim(a, b, height, width, win) {
    win = win || 3;
    const L = 1;
    const C1 = (0.01 * L) ** 2;
    const C2 = (0.03 * L) ** 2;
    const n = win * win;

    let total = 0;
    let count = 0;

    for (let y0 = 0; y0 <= height - win; y0++) {
      for (let x0 = 0; x0 <= width - win; x0++) {
        let sumA = 0, sumB = 0;
        for (let dy = 0; dy < win; dy++) {
          for (let dx = 0; dx < win; dx++) {
            const idx = (y0 + dy) * width + (x0 + dx);
            sumA += a[idx];
            sumB += b[idx];
          }
        }
        const muA = sumA / n;
        const muB = sumB / n;

        let varA = 0, varB = 0, covAB = 0;
        for (let dy = 0; dy < win; dy++) {
          for (let dx = 0; dx < win; dx++) {
            const idx = (y0 + dy) * width + (x0 + dx);
            const da = a[idx] - muA;
            const db = b[idx] - muB;
            varA += da * da;
            varB += db * db;
            covAB += da * db;
          }
        }
        varA /= (n - 1 || 1);
        varB /= (n - 1 || 1);
        covAB /= (n - 1 || 1);

        const num = (2 * muA * muB + C1) * (2 * covAB + C2);
        const den = (muA * muA + muB * muB + C1) * (varA + varB + C2);
        total += num / den;
        count++;
      }
    }

    const mean = count ? total / count : 1;
    return Math.max(0, Math.min(1, mean));
  }

  // Score a generated sample against every entry in a training set (array
  // of { file, text, bin }, where bin is a precomputed textToBinary result —
  // see app.js). Returns the best (highest-SSIM) match.
  function bestMatch(genBin, trainSet, height, width, win) {
    let best = { index: -1, file: null, text: null, ssim: -1, hamming: 0 };
    for (let i = 0; i < trainSet.length; i++) {
      const entry = trainSet[i];
      const s = ssim(genBin, entry.bin, height, width, win);
      if (s > best.ssim) {
        const h = hammingSimilarity(genBin, entry.bin);
        best = { index: i, file: entry.file, text: entry.text, ssim: s, hamming: h };
      }
    }
    return best;
  }

  root.AsciiSimilarity = { textToBinary, hammingSimilarity, ssim, bestMatch };
})(typeof window !== 'undefined' ? window : this);
