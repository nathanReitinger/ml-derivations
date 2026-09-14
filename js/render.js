(function (root) {
  // Keep in sync with the accent colors in css/style.css (--violet, --teal,
  // --ink, --coral, --amber). This dataset's vocab is just 2 characters, so
  // in practice only PALETTE[0] (violet) ever gets used as the "ink" color —
  // the rest exist for any vocab with more than one foreground character.
  const PALETTE = ['#6C63FF', '#14B8A6', '#15161D', '#FF6161', '#FFB020'];

  // Figure out which vocab character is "background" (most frequent across
  // a sample of texts) so it can be rendered as empty space rather than a
  // color, and assign the rest of the vocab a stable palette.
  function buildCharStyle(vocab, sampleTexts) {
    const counts = new Map(vocab.map(c => [c, 0]));
    for (const t of sampleTexts) {
      for (const ch of t) {
        if (counts.has(ch)) counts.set(ch, counts.get(ch) + 1);
      }
    }
    let bg = vocab[0];
    let bgCount = -1;
    for (const [ch, n] of counts) {
      if (n > bgCount) { bgCount = n; bg = ch; }
    }
    const fg = vocab.filter(c => c !== bg);
    const colorOf = new Map();
    fg.forEach((ch, i) => colorOf.set(ch, PALETTE[i % PALETTE.length]));
    return { bg, colorOf };
  }

  function rowsOf(text, height, width) {
    const rows = text.split('\n');
    return rows;
  }

  function renderGridToCanvas(canvas, text, { height, width, cell, style, voidColor }) {
    canvas.width = width * cell;
    canvas.height = height * cell;
    const ctx = canvas.getContext('2d');
    // Transparent by default so the CSS chip/card behind the canvas shows
    // through — the light theme controls the "paper" color in one place
    // (style.css) instead of duplicating it here.
    if (voidColor) {
      ctx.fillStyle = voidColor;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    const rows = rowsOf(text, height, width);
    for (let r = 0; r < height; r++) {
      const row = rows[r] || '';
      for (let c = 0; c < width; c++) {
        const ch = row[c];
        if (ch === undefined || ch === style.bg) continue;
        ctx.fillStyle = style.colorOf.get(ch) || PALETTE[0];
        ctx.fillRect(c * cell, r * cell, cell, cell);
      }
    }
  }

  // Returns a mask of differing cell count, for the "0 differing cells" proof.
  function diffCount(textA, textB) {
    const a = textA.replace(/\n/g, '');
    const b = textB.replace(/\n/g, '');
    let n = 0;
    const len = Math.max(a.length, b.length);
    for (let i = 0; i < len; i++) if (a[i] !== b[i]) n++;
    return n;
  }

  root.AsciiRender = { buildCharStyle, renderGridToCanvas, diffCount };
})(typeof window !== 'undefined' ? window : this);
