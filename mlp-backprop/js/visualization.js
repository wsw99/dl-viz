/**
 * visualization.js — MLP network canvas renderer (no external chart library)
 * All drawing uses the native Canvas 2D API only.
 */
const Viz = (() => {

  // ── Canvas & dimensions ──────────────────────────────────────
  let canvas, ctx, W, H;
  const NODE_R = 26;

  // ── Loss chart (drawn on a separate small canvas) ────────────
  let lossCanvas, lossCtx;
  const lossHistory = [];

  // ── Node positions (computed once at init) ───────────────────
  const nodes = {};
  const LAYER_X = [];

  // ─────────────────────────────────────────────────────────────
  // INIT
  // ─────────────────────────────────────────────────────────────
  function initCanvas() {
    canvas = document.getElementById('network-canvas');
    ctx    = canvas.getContext('2d');

    // Fixed size from window — NO resize listener of any kind
    W = Math.max(320, window.innerWidth  - 380);
    H = Math.max(300, window.innerHeight - 120);
    canvas.width  = W;
    canvas.height = H;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';

    computeLayout();
  }

  function initLossChart() {
    lossCanvas = document.getElementById('loss-chart');
    if (!lossCanvas) return;
    lossCanvas.width  = 280;
    lossCanvas.height = 100;
    lossCanvas.style.width  = '280px';
    lossCanvas.style.height = '100px';
    lossCtx = lossCanvas.getContext('2d');
    drawLossCurve(); // draw empty axes
  }

  // ─────────────────────────────────────────────────────────────
  // LAYOUT
  // ─────────────────────────────────────────────────────────────
  function buildEdges() { /* edges derived on-the-fly from node ids */ }

  function computeLayout() {
    const layers = [
      ['i0', 'i1'],
      ['h0', 'h1', 'h2'],
      ['o0'],
    ];
    const xStart = W * 0.18, xEnd = W * 0.82;
    const xStep  = (xEnd - xStart) / (layers.length - 1);

    layers.forEach((ids, li) => {
      LAYER_X[li] = xStart + li * xStep;
      ids.forEach((id, ni) => {
        const totalH = (ids.length - 1) * 90;
        nodes[id] = {
          x: LAYER_X[li],
          y: H / 2 - totalH / 2 + ni * 90,
        };
      });
    });
    // Virtual loss node below output
    nodes['loss'] = { x: LAYER_X[2], y: nodes['o0'].y + 85 };
  }

  // All directed edges
  const ALL_EDGES = [
    ['i0','h0'],['i0','h1'],['i0','h2'],
    ['i1','h0'],['i1','h1'],['i1','h2'],
    ['h0','o0'],['h1','o0'],['h2','o0'],
  ];

  // ─────────────────────────────────────────────────────────────
  // MAIN RENDER (pure function — called by StepEngine)
  // ─────────────────────────────────────────────────────────────
  function render(step, idx, _total) {
    if (!canvas) return;
    ctx.clearRect(0, 0, W, H);

    const { activations, weights, gradients, deltas,
            highlightEdges, highlightNodes, phase, flowDir, loss } = step;

    const hlAll = highlightEdges === 'all';

    // ── Edges ──────────────────────────────────────────────────
    ALL_EDGES.forEach(([from, to]) => {
      const edgeId = from + to;
      const isHL   = hlAll || (Array.isArray(highlightEdges) && highlightEdges.includes(edgeId));
      const nf = nodes[from], nt = nodes[to];

      let color = '#2a2d45';
      let lw    = 1.5;

      if (weights) {
        const w = edgeWeight(weights, from, to);
        if (w !== null) {
          const intensity = Math.min(Math.abs(w) / 1.5, 1);
          color = w >= 0
            ? `rgba(167,139,250,${0.15 + intensity * 0.55})`
            : `rgba(96,165,250,${0.15 + intensity * 0.55})`;
          lw = 1 + intensity * 3;
        }
      }

      if ((phase === 'backward') && gradients) {
        const g = edgeGrad(gradients, from, to);
        if (g !== null) {
          const intensity = Math.min(Math.abs(g) / 0.8, 1);
          color = g >= 0
            ? `rgba(248,113,113,${0.3 + intensity * 0.7})`
            : `rgba(96,165,250,${0.3 + intensity * 0.7})`;
          lw = 1.5 + intensity * 3;
        }
      }

      if (phase === 'update' && deltas) {
        const d = edgeDelta(deltas, from, to);
        if (d !== null) {
          const intensity = Math.min(Math.abs(d) / 0.3, 1);
          color = d >= 0
            ? `rgba(248,113,113,${0.4 + intensity * 0.6})`
            : `rgba(96,165,250,${0.4 + intensity * 0.6})`;
          lw = 1.5 + intensity * 3.5;
        }
      }

      ctx.beginPath();
      ctx.moveTo(nf.x, nf.y);
      ctx.lineTo(nt.x, nt.y);
      ctx.strokeStyle = color;
      ctx.lineWidth   = isHL ? lw + 1.5 : lw;
      ctx.globalAlpha = isHL ? 1 : 0.7;
      ctx.stroke();
      ctx.globalAlpha = 1;

      // Gradient flow arrow on highlighted edges during backward
      if (isHL && (phase === 'backward' || flowDir === 'backward')) {
        drawArrowHead(nf.x, nf.y, nt.x, nt.y, '#f87171');
      }

      // Weight label on highlighted edges
      if (isHL && weights) {
        const w = edgeWeight(weights, from, to);
        if (w !== null) {
          const mx = (nf.x + nt.x) / 2;
          const my = (nf.y + nt.y) / 2 - 7;
          ctx.fillStyle = '#cbd5e1';
          ctx.font = '10px Consolas';
          ctx.textAlign = 'center';
          ctx.fillText(w.toFixed(3), mx, my);
        }
      }
    });

    // ── Nodes ──────────────────────────────────────────────────
    const allNodeIds = ['i0','i1','h0','h1','h2','o0'];
    allNodeIds.forEach(id => {
      const n    = nodes[id];
      const isHL = Array.isArray(highlightNodes) && highlightNodes.includes(id);
      const aval = nodeActivation(activations, id);

      // Glow ring
      if (isHL) {
        ctx.beginPath();
        ctx.arc(n.x, n.y, NODE_R + 9, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(167,139,250,0.12)';
        ctx.fill();
      }

      // Node fill
      ctx.beginPath();
      ctx.arc(n.x, n.y, NODE_R, 0, Math.PI * 2);
      ctx.fillStyle   = aval !== null ? activationFill(aval) : '#181b2e';
      ctx.strokeStyle = isHL ? '#a78bfa' : '#3d4266';
      ctx.lineWidth   = isHL ? 2.5 : 1.5;
      ctx.fill();
      ctx.stroke();

      // Value or label text
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      if (aval !== null) {
        ctx.fillStyle = '#fff';
        ctx.font      = '11px Consolas';
        ctx.fillText(aval.toFixed(3), n.x, n.y);
      } else {
        ctx.fillStyle = '#475569';
        ctx.font      = 'bold 10px Segoe UI';
        ctx.fillText(id.toUpperCase(), n.x, n.y);
      }

      // Gradient badge below node (backward / update phases)
      if ((phase === 'backward' || phase === 'update') && gradients) {
        const g = nodeGrad(gradients, id);
        if (g !== null) {
          ctx.fillStyle = g > 0 ? '#f87171' : '#60a5fa';
          ctx.font = '9px Consolas';
          ctx.fillText(`∂=${g.toFixed(3)}`, n.x, n.y + NODE_R + 12);
        }
      }
    });

    // Loss node
    if (loss !== null && loss !== undefined) {
      const ln   = nodes['loss'];
      const isHL = Array.isArray(highlightNodes) && highlightNodes.includes('loss');
      ctx.beginPath();
      ctx.arc(ln.x, ln.y, NODE_R - 4, 0, Math.PI * 2);
      ctx.fillStyle   = isHL ? '#5b21b6' : '#1e2130';
      ctx.strokeStyle = isHL ? '#a78bfa' : '#3d4266';
      ctx.lineWidth   = 1.5;
      ctx.fill(); ctx.stroke();
      ctx.fillStyle    = '#e2e8f0';
      ctx.font         = '10px Consolas';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`L=${loss.toFixed(3)}`, ln.x, ln.y);

      // Thin line from o0 to loss node
      ctx.beginPath();
      ctx.moveTo(nodes['o0'].x, nodes['o0'].y + NODE_R);
      ctx.lineTo(ln.x, ln.y - (NODE_R - 4));
      ctx.strokeStyle = isHL ? '#a78bfa' : '#2a2d45';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // ── Layer labels ───────────────────────────────────────────
    const lbls = ['Input', 'Hidden (ReLU)', 'Output (σ)'];
    lbls.forEach((lbl, i) => {
      ctx.fillStyle    = '#374151';
      ctx.font         = '11px Segoe UI';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(lbl, LAYER_X[i], H - 14);
    });

    // ── ReLU mini-chart (Step 3) ────────────────────────────────
    if (step.showRelu) drawReluInset(step.reluInputs);

    // ── Sigmoid mini-chart (Step 5) ─────────────────────────────
    if (step.showSigmoid) drawSigmoidInset(step.sigmoidInput);

    // ── Sidebar updates ─────────────────────────────────────────
    setCaption(step.caption);
    setPseudocode(step.codeLine);
    setParams(step);
    appendLoss(idx, loss);
  }

  // ─────────────────────────────────────────────────────────────
  // INSET CHARTS (drawn on the main network canvas)
  // ─────────────────────────────────────────────────────────────
  function drawReluInset(zVals) {
    const cx = LAYER_X[1], cy = 42, cw = 90, ch = 55;
    insetBox(cx - cw/2, cy - ch/2, cw, ch, 'ReLU(z)');

    ctx.beginPath(); ctx.strokeStyle = '#34d399'; ctx.lineWidth = 1.5;
    for (let px = 0; px <= cw; px++) {
      const z  = -2 + (px / cw) * 4;
      const sy = cy + ch/2 - (Math.max(0, z) / 2) * ch * 0.9;
      px === 0 ? ctx.moveTo(cx - cw/2 + px, sy) : ctx.lineTo(cx - cw/2 + px, sy);
    }
    ctx.stroke();

    (zVals || []).forEach((z, k) => {
      const px = ((z + 2) / 4) * cw;
      const a  = Math.max(0, z);
      const sy = cy + ch/2 - (a / 2) * ch * 0.9;
      ctx.beginPath();
      ctx.arc(cx - cw/2 + px, sy, 3.5, 0, Math.PI*2);
      ctx.fillStyle = ['#f87171','#60a5fa','#34d399'][k] || '#e2e8f0';
      ctx.fill();
    });
  }

  function drawSigmoidInset(zIn) {
    const cx = LAYER_X[2], cy = 42, cw = 90, ch = 55;
    insetBox(cx - cw/2, cy - ch/2, cw, ch, 'σ(z)');

    ctx.beginPath(); ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5;
    for (let px = 0; px <= cw; px++) {
      const z  = -4 + (px / cw) * 8;
      const s  = 1 / (1 + Math.exp(-z));
      const sy = cy + ch/2 - s * ch * 0.85 - ch * 0.05;
      px === 0 ? ctx.moveTo(cx - cw/2 + px, sy) : ctx.lineTo(cx - cw/2 + px, sy);
    }
    ctx.stroke();

    if (zIn !== null && zIn !== undefined) {
      const px  = ((zIn + 4) / 8) * cw;
      const s   = 1 / (1 + Math.exp(-zIn));
      const sx  = cx - cw/2 + px;
      const sy  = cy + ch/2 - s * ch * 0.85 - ch * 0.05;
      ctx.beginPath(); ctx.arc(sx, sy, 4, 0, Math.PI*2);
      ctx.fillStyle = '#f59e0b'; ctx.fill();
      ctx.setLineDash([3, 3]); ctx.strokeStyle = '#f59e0b55'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx, cy + ch/2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(cx - cw/2, sy); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function insetBox(x, y, w, h, label) {
    ctx.fillStyle   = '#1a1d2e';
    ctx.strokeStyle = '#2d3148';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 4);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle    = '#475569';
    ctx.font         = '9px Segoe UI';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(label, x + w/2, y - 3);
  }

  // ─────────────────────────────────────────────────────────────
  // LOSS CURVE (separate small canvas, fixed pixel size)
  // ─────────────────────────────────────────────────────────────
  function appendLoss(idx, loss) {
    if (idx === 0) lossHistory.length = 0;   // reset on replay
    if (loss !== null && loss !== undefined) {
      const last = lossHistory[lossHistory.length - 1];
      if (last !== loss) lossHistory.push(loss);
    }
    drawLossCurve();
  }

  function drawLossCurve() {
    if (!lossCtx) return;
    const lw = lossCanvas.width, lh = lossCanvas.height;
    const pad = { t: 8, r: 8, b: 22, l: 36 };
    const pw = lw - pad.l - pad.r;
    const ph = lh - pad.t - pad.b;

    lossCtx.clearRect(0, 0, lw, lh);

    // Background
    lossCtx.fillStyle = '#1a1d2e';
    lossCtx.fillRect(0, 0, lw, lh);

    // Axes
    lossCtx.strokeStyle = '#2d3148';
    lossCtx.lineWidth   = 1;
    lossCtx.beginPath();
    lossCtx.moveTo(pad.l, pad.t);
    lossCtx.lineTo(pad.l, pad.t + ph);
    lossCtx.lineTo(pad.l + pw, pad.t + ph);
    lossCtx.stroke();

    // Axis labels
    lossCtx.fillStyle    = '#475569';
    lossCtx.font         = '9px Segoe UI';
    lossCtx.textAlign    = 'right';
    lossCtx.textBaseline = 'middle';
    lossCtx.fillText('Step', pad.l + pw, pad.t + ph + 13);

    if (lossHistory.length < 2) {
      lossCtx.fillStyle    = '#374151';
      lossCtx.font         = '10px Segoe UI';
      lossCtx.textAlign    = 'center';
      lossCtx.textBaseline = 'middle';
      lossCtx.fillText('损失将在 Step 6 出现', pad.l + pw/2, pad.t + ph/2);
      return;
    }

    const maxL = Math.max(...lossHistory) * 1.1;
    const minL = 0;

    // Y ticks
    lossCtx.fillStyle    = '#475569';
    lossCtx.font         = '8px Consolas';
    lossCtx.textAlign    = 'right';
    lossCtx.textBaseline = 'middle';
    for (let t = 0; t <= 4; t++) {
      const val = maxL * (1 - t/4);
      const sy  = pad.t + (t/4) * ph;
      lossCtx.fillText(val.toFixed(2), pad.l - 3, sy);
      lossCtx.strokeStyle = '#21253a';
      lossCtx.lineWidth   = 0.5;
      lossCtx.beginPath();
      lossCtx.moveTo(pad.l, sy);
      lossCtx.lineTo(pad.l + pw, sy);
      lossCtx.stroke();
    }

    // Loss line
    lossCtx.beginPath();
    lossCtx.strokeStyle = '#a78bfa';
    lossCtx.lineWidth   = 1.8;
    lossCtx.lineJoin    = 'round';
    lossHistory.forEach((v, i) => {
      const sx = pad.l + (i / (lossHistory.length - 1 || 1)) * pw;
      const sy = pad.t + ph - ((v - minL) / (maxL - minL)) * ph;
      i === 0 ? lossCtx.moveTo(sx, sy) : lossCtx.lineTo(sx, sy);
    });
    lossCtx.stroke();

    // Dots
    lossHistory.forEach((v, i) => {
      const sx = pad.l + (i / (lossHistory.length - 1 || 1)) * pw;
      const sy = pad.t + ph - ((v - minL) / (maxL - minL)) * ph;
      lossCtx.beginPath();
      lossCtx.arc(sx, sy, 3, 0, Math.PI*2);
      lossCtx.fillStyle = '#a78bfa';
      lossCtx.fill();
      // Value label
      lossCtx.fillStyle    = '#c4b5fd';
      lossCtx.font         = '8px Consolas';
      lossCtx.textAlign    = 'center';
      lossCtx.textBaseline = 'bottom';
      lossCtx.fillText(v.toFixed(3), sx, sy - 4);
    });
  }

  // ─────────────────────────────────────────────────────────────
  // SIDEBAR DOM UPDATES
  // ─────────────────────────────────────────────────────────────
  function setCaption(text) {
    const el = document.getElementById('caption-text');
    if (el && text) el.textContent = text;
  }

  const PSEUDOCODE = [
    '// Initialize weights W1, b1, W2, b2',
    'x = [x1, x2]  // input sample',
    'z1 = W1 · x + b1  // hidden pre-act.',
    'a1 = ReLU(z1)  // hidden activation',
    'z2 = W2 · a1 + b2  // output pre-act.',
    'ŷ  = σ(z2)  // output activation',
    'L  = BCE(y, ŷ)  // compute loss',
    '∂L/∂ŷ = ŷ − y  // output gradient',
    '∂L/∂W2 = δ2 · a1ᵀ',
    '∂L/∂a1 = W2ᵀ · δ2',
    '∂L/∂z1 = ∂L/∂a1 ⊙ ReLU′(z1)',
    '∂L/∂W1 = δ1 · xᵀ',
    'W ← W − lr · ∂L/∂W  // update',
    '// Verify: re-run forward pass',
  ];

  function setPseudocode(activeLine) {
    const box = document.getElementById('pseudo-lines');
    if (!box) return;
    box.innerHTML = PSEUDOCODE.map((line, i) => {
      const cls = i === activeLine ? 'line active'
                : i <  activeLine  ? 'line done'
                :                    'line';
      return `<span class="${cls}">${String(i).padStart(2)} ${line}</span>`;
    }).join('');
  }

  function setParams(step) {
    const box = document.getElementById('param-values');
    if (!box) return;
    const { activations, gradients, loss, weights } = step;
    const rows = [];

    if (loss != null)
      rows.push(['Loss (BCE)', loss.toFixed(5), loss < 0.3 ? 'good' : '']);
    if (activations?.a2?.[0] != null)
      rows.push(['Output ŷ', activations.a2[0].toFixed(4), '']);
    if (gradients?.dL_da2?.[0] != null)
      rows.push(['∂L/∂ŷ', gradients.dL_da2[0].toFixed(4),
                 gradients.dL_da2[0] > 0 ? 'pos' : 'neg']);
    if (weights?.w2?.[0])
      weights.w2[0].forEach((w, k) => rows.push([`W₂[${k}]`, w.toFixed(4), '']));

    box.innerHTML = rows.map(([lbl, val, cls]) =>
      `<div class="param-item">
         <div class="param-label">${lbl}</div>
         <div class="param-value ${cls}">${val}</div>
       </div>`
    ).join('');
  }

  // ─────────────────────────────────────────────────────────────
  // DRAWING HELPERS
  // ─────────────────────────────────────────────────────────────
  function drawArrowHead(x1, y1, x2, y2, color) {
    // Arrow points in backward direction (from output side toward input)
    const angle = Math.atan2(y1 - y2, x1 - x2);
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const sz = 7;
    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.lineTo(mx + sz * Math.cos(angle - 0.45), my + sz * Math.sin(angle - 0.45));
    ctx.lineTo(mx + sz * Math.cos(angle + 0.45), my + sz * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
  }

  function activationFill(a) {
    const t = Math.min(1, Math.max(0, a));
    return `rgb(${Math.round(50 + t*170)},${Math.round(30)},${Math.round(200 - t*160)})`;
  }

  // ─────────────────────────────────────────────────────────────
  // WEIGHT / GRADIENT LOOKUPS
  // ─────────────────────────────────────────────────────────────
  const INPUT_IDS  = ['i0','i1'];
  const HIDDEN_IDS = ['h0','h1','h2'];

  function edgeWeight(w, from, to) {
    if (!w) return null;
    if (INPUT_IDS.includes(from) && HIDDEN_IDS.includes(to))
      return w.w1?.[HIDDEN_IDS.indexOf(to)]?.[INPUT_IDS.indexOf(from)] ?? null;
    if (HIDDEN_IDS.includes(from) && to === 'o0')
      return w.w2?.[0]?.[HIDDEN_IDS.indexOf(from)] ?? null;
    return null;
  }

  function edgeGrad(g, from, to) {
    if (!g) return null;
    if (HIDDEN_IDS.includes(from) && to === 'o0')
      return g.dL_dw2?.[0]?.[HIDDEN_IDS.indexOf(from)] ?? null;
    if (INPUT_IDS.includes(from) && HIDDEN_IDS.includes(to))
      return g.dL_dw1?.[HIDDEN_IDS.indexOf(to)]?.[INPUT_IDS.indexOf(from)] ?? null;
    return null;
  }

  function edgeDelta(d, from, to) {
    if (!d) return null;
    if (HIDDEN_IDS.includes(from) && to === 'o0')
      return d.deltaW2?.[0]?.[HIDDEN_IDS.indexOf(from)] ?? null;
    if (INPUT_IDS.includes(from) && HIDDEN_IDS.includes(to))
      return d.deltaW1?.[HIDDEN_IDS.indexOf(to)]?.[INPUT_IDS.indexOf(from)] ?? null;
    return null;
  }

  function nodeActivation(a, id) {
    if (!a) return null;
    const map = { i0: a.a0?.[0], i1: a.a0?.[1],
                  h0: a.a1?.[0], h1: a.a1?.[1], h2: a.a1?.[2],
                  o0: a.a2?.[0] };
    const v = map[id];
    return v !== undefined && v !== null ? v : null;
  }

  function nodeGrad(g, id) {
    if (!g) return null;
    const map = { o0: g.dL_da2?.[0],
                  h0: g.dL_da1?.[0], h1: g.dL_da1?.[1], h2: g.dL_da1?.[2] };
    const v = map[id];
    return v !== undefined && v !== null ? v : null;
  }

  // ─────────────────────────────────────────────────────────────
  return { initCanvas, buildEdges, initLossChart, render };
})();
