/**
 * algorithm.js — MLP forward + backward pass, pure JS.
 * Produces an array of step snapshots consumed by the renderer.
 *
 * Network: 2 → 3(ReLU) → 1(Sigmoid)   Loss: Binary Cross-Entropy
 */

// ── Math helpers ──────────────────────────────────────────────
const relu    = z => Math.max(0, z);
const reluD   = z => z > 0 ? 1 : 0;
const sigmoid = z => 1 / (1 + Math.exp(-z));
const bce     = (y, yhat) => -(y * Math.log(yhat + 1e-9) + (1 - y) * Math.log(1 - yhat + 1e-9));

// Deep-clone a weight object
function cloneW(w) {
  return {
    w1: w.w1.map(r => [...r]),
    b1: [...w.b1],
    w2: w.w2.map(r => [...r]),
    b2: [...w.b2],
  };
}

// Matrix-vector multiply: M [rows×cols] · v [cols] → out [rows]
function matVec(M, v) {
  return M.map(row => row.reduce((s, mij, j) => s + mij * v[j], 0));
}

/**
 * buildSteps(data) → steps[]
 * Each step snapshot has the shape described below.
 */
function buildSteps(data) {
  const steps = [];
  const { samples, arch, lr, initWeights, demoSampleIdx } = data;
  const sample = samples[demoSampleIdx]; // {x:[0,1], y:1}

  // Working copy of weights (mutated during the demo)
  let W = cloneW(initWeights);

  // ── Helper: push a snapshot ──────────────────────────────────
  function snap(phase, substep, extra = {}) {
    steps.push({
      phase,      // 'init'|'forward'|'loss'|'backward'|'update'|'converge'
      substep,    // string key for pseudocode highlight + caption
      weights: cloneW(W),
      ...extra,
    });
  }

  // ── STEP 0 — Initial network ──────────────────────────────────
  snap('init', 'init', {
    caption: '初始化网络。网络结构：2个输入节点 → 3个隐藏节点（ReLU）→ 1个输出节点（Sigmoid）。权重已随机初始化，我们将对样本 x=[0,1], y=1 进行一次完整的前向+反向传播。',
    codeLine: 0,
    activations: { a0: [0, 1], z1: null, a1: null, z2: null, a2: null },
    gradients: null,
    loss: null,
    highlightEdges: [],
    highlightNodes: [],
  });

  // ── STEP 1 — Show input ───────────────────────────────────────
  snap('forward', 'input', {
    caption: '输入层：将样本 x₁=0, x₂=1 送入网络。输入节点激活值 a₀ = x。',
    codeLine: 1,
    activations: { a0: [0, 1], z1: null, a1: null, z2: null, a2: null },
    gradients: null,
    loss: null,
    highlightNodes: ['i0', 'i1'],
    highlightEdges: [],
  });

  // ── STEP 2 — Forward: hidden pre-activation z1 ───────────────
  const a0 = [0, 1]; // x
  const z1 = matVec(W.w1, a0).map((z, k) => z + W.b1[k]);

  snap('forward', 'z1', {
    caption: `隐藏层线性变换：z₁ = W₁·x + b₁。\n对每个隐藏节点 k：z₁ₖ = Σ w₁ₖⱼ·xⱼ + b₁ₖ\nz₁ = [${z1.map(v=>v.toFixed(3)).join(', ')}]`,
    codeLine: 2,
    activations: { a0, z1: [...z1], a1: null, z2: null, a2: null },
    gradients: null,
    loss: null,
    highlightNodes: ['h0','h1','h2'],
    highlightEdges: ['i0h0','i0h1','i0h2','i1h0','i1h1','i1h2'],
  });

  // ── STEP 3 — Forward: hidden activation a1 (ReLU) ────────────
  const a1 = z1.map(relu);

  snap('forward', 'a1', {
    caption: `ReLU 激活：a₁ = ReLU(z₁) = max(0, z₁)。\na₁ = [${a1.map(v=>v.toFixed(3)).join(', ')}]\n负值被截断为 0，只有正值能"通过"。`,
    codeLine: 3,
    activations: { a0, z1: [...z1], a1: [...a1], z2: null, a2: null },
    gradients: null,
    loss: null,
    highlightNodes: ['h0','h1','h2'],
    highlightEdges: [],
    showRelu: true,
    reluInputs: [...z1],
  });

  // ── STEP 4 — Forward: output pre-activation z2 ───────────────
  const z2 = matVec(W.w2, a1).map((z, k) => z + W.b2[k]);

  snap('forward', 'z2', {
    caption: `输出层线性变换：z₂ = W₂·a₁ + b₂。\nz₂ = ${z2[0].toFixed(4)}`,
    codeLine: 4,
    activations: { a0, z1: [...z1], a1: [...a1], z2: [...z2], a2: null },
    gradients: null,
    loss: null,
    highlightNodes: ['o0'],
    highlightEdges: ['h0o0','h1o0','h2o0'],
  });

  // ── STEP 5 — Forward: output activation a2 (Sigmoid) ─────────
  const a2 = z2.map(sigmoid);
  const yhat = a2[0];
  const y = sample.y;

  snap('forward', 'a2', {
    caption: `Sigmoid 激活：ŷ = σ(z₂) = 1/(1+e^−z₂) = ${yhat.toFixed(4)}。\n真实标签 y = ${y}。ŷ 越接近 y 越好。`,
    codeLine: 5,
    activations: { a0, z1: [...z1], a1: [...a1], z2: [...z2], a2: [...a2] },
    gradients: null,
    loss: null,
    highlightNodes: ['o0'],
    highlightEdges: [],
    showSigmoid: true,
    sigmoidInput: z2[0],
  });

  // ── STEP 6 — Compute loss ─────────────────────────────────────
  const loss = bce(y, yhat);

  snap('loss', 'loss', {
    caption: `计算损失（Binary Cross-Entropy）：\nL = −[y·log(ŷ) + (1−y)·log(1−ŷ)]\nL = ${loss.toFixed(4)}\n损失越小说明预测越准确。`,
    codeLine: 6,
    activations: { a0, z1: [...z1], a1: [...a1], z2: [...z2], a2: [...a2] },
    gradients: null,
    loss,
    highlightNodes: ['loss'],
    highlightEdges: [],
  });

  // ── STEP 7 — Backward: output layer gradient ─────────────────
  // dL/dŷ = (ŷ - y) / (ŷ(1-ŷ))  [for BCE + Sigmoid combined = ŷ - y]
  const dL_da2 = [yhat - y];                  // combined BCE+Sigmoid gradient
  const dL_dz2 = [...dL_da2];                 // because dσ/dz * dL/da = dL_da2 already

  snap('backward', 'grad_output', {
    caption: `反向传播起点：计算输出层梯度。\n∂L/∂ŷ = ŷ − y = ${dL_da2[0].toFixed(4)}\n（BCE损失 + Sigmoid 激活的联合梯度，化简后等于 ŷ−y）`,
    codeLine: 7,
    activations: { a0, z1, a1, z2, a2 },
    gradients: { dL_da2: [...dL_da2], dL_dz2: [...dL_dz2], dL_dw2: null, dL_da1: null, dL_dz1: null, dL_dw1: null },
    loss,
    highlightNodes: ['o0'],
    highlightEdges: [],
    flowDir: 'backward',
  });

  // ── STEP 8 — Backward: grad w.r.t. W2 ───────────────────────
  // dL/dW2[0][k] = dL/dz2[0] * a1[k]
  const dL_dw2 = [a1.map(ak => dL_dz2[0] * ak)];
  const dL_db2 = [...dL_dz2];

  snap('backward', 'grad_w2', {
    caption: `计算输出层权重梯度：\n∂L/∂W₂ₖ = (∂L/∂z₂) · a₁ₖ\n∂L/∂W₂ = [${dL_dw2[0].map(v=>v.toFixed(4)).join(', ')}]\n∂L/∂b₂ = ${dL_db2[0].toFixed(4)}`,
    codeLine: 8,
    activations: { a0, z1, a1, z2, a2 },
    gradients: { dL_da2, dL_dz2, dL_dw2: dL_dw2.map(r=>[...r]), dL_db2: [...dL_db2], dL_da1: null, dL_dz1: null, dL_dw1: null },
    loss,
    highlightEdges: ['h0o0','h1o0','h2o0'],
    flowDir: 'backward',
  });

  // ── STEP 9 — Backward: propagate to hidden layer ─────────────
  // dL/da1[k] = dL/dz2[0] * W2[0][k]
  const dL_da1 = W.w2[0].map(wk => dL_dz2[0] * wk);

  snap('backward', 'grad_a1', {
    caption: `将梯度传回隐藏层：\n∂L/∂a₁ₖ = (∂L/∂z₂) · W₂ₖ（链式法则）\n∂L/∂a₁ = [${dL_da1.map(v=>v.toFixed(4)).join(', ')}]\n梯度沿红色箭头从右向左流动。`,
    codeLine: 9,
    activations: { a0, z1, a1, z2, a2 },
    gradients: { dL_da2, dL_dz2, dL_dw2, dL_db2, dL_da1: [...dL_da1], dL_dz1: null, dL_dw1: null },
    loss,
    highlightEdges: ['h0o0','h1o0','h2o0'],
    flowDir: 'backward',
  });

  // ── STEP 10 — Backward: ReLU gradient ────────────────────────
  // dL/dz1[k] = dL/da1[k] * ReLU'(z1[k])
  const dL_dz1 = dL_da1.map((g, k) => g * reluD(z1[k]));

  snap('backward', 'grad_relu', {
    caption: `ReLU 反向传播（截断梯度）：\n∂L/∂z₁ₖ = ∂L/∂a₁ₖ · ReLU′(z₁ₖ)\nReLU′(z) = 1 (z>0) 或 0 (z≤0)\n∂L/∂z₁ = [${dL_dz1.map(v=>v.toFixed(4)).join(', ')}]\nz₁ₖ≤0 的节点梯度被截断为 0（"死亡ReLU"）。`,
    codeLine: 10,
    activations: { a0, z1, a1, z2, a2 },
    gradients: { dL_da2, dL_dz2, dL_dw2, dL_db2, dL_da1, dL_dz1: [...dL_dz1], dL_dw1: null },
    loss,
    highlightNodes: ['h0','h1','h2'],
    flowDir: 'backward',
  });

  // ── STEP 11 — Backward: grad w.r.t. W1 ───────────────────────
  // dL/dW1[k][j] = dL/dz1[k] * a0[j]
  const dL_dw1 = dL_dz1.map(g => a0.map(xj => g * xj));
  const dL_db1 = [...dL_dz1];

  snap('backward', 'grad_w1', {
    caption: `计算输入层权重梯度：\n∂L/∂W₁ₖⱼ = (∂L/∂z₁ₖ) · xⱼ\n所有梯度已计算完毕，准备更新权重。`,
    codeLine: 11,
    activations: { a0, z1, a1, z2, a2 },
    gradients: { dL_da2, dL_dz2, dL_dw2, dL_db2, dL_da1, dL_dz1, dL_dw1: dL_dw1.map(r=>[...r]), dL_db1: [...dL_db1] },
    loss,
    highlightEdges: ['i0h0','i0h1','i0h2','i1h0','i1h1','i1h2'],
    flowDir: 'backward',
  });

  // ── STEP 12 — Update weights ──────────────────────────────────
  // Store deltas for visualization
  const deltaW1 = dL_dw1.map(r => r.map(g => -lr * g));
  const deltaB1 = dL_db1.map(g => -lr * g);
  const deltaW2 = dL_dw2.map(r => r.map(g => -lr * g));
  const deltaB2 = dL_db2.map(g => -lr * g);

  // Apply updates
  W.w1 = W.w1.map((row, k) => row.map((w, j) => w + deltaW1[k][j]));
  W.b1 = W.b1.map((b, k) => b + deltaB1[k]);
  W.w2 = W.w2.map((row, k) => row.map((w, j) => w + deltaW2[k][j]));
  W.b2 = W.b2.map((b, k) => b + deltaB2[k]);

  snap('update', 'update', {
    caption: `权重更新：W ← W − lr · ∂L/∂W，lr = ${lr}\n边的颜色变化：红色=权重增大，蓝色=权重减小。\n边的粗细变化：粗=更新幅度大。`,
    codeLine: 12,
    activations: { a0, z1, a1, z2, a2 },
    gradients: { dL_dw1, dL_db1, dL_dw2, dL_db2 },
    deltas: { deltaW1, deltaB1, deltaW2, deltaB2 },
    loss,
    highlightEdges: 'all',
    flowDir: 'update',
  });

  // ── STEP 13 — Verify new loss (forward pass with new W) ───────
  const z1new  = matVec(W.w1, a0).map((z, k) => z + W.b1[k]);
  const a1new  = z1new.map(relu);
  const z2new  = matVec(W.w2, a1new).map((z, k) => z + W.b2[k]);
  const a2new  = z2new.map(sigmoid);
  const lossNew = bce(y, a2new[0]);

  snap('converge', 'result', {
    caption: `更新后验证：对同一样本重新前向传播。\n新损失 L = ${lossNew.toFixed(4)}（原来 ${loss.toFixed(4)}）\n损失下降了 ${(loss - lossNew).toFixed(4)}。\n重复这个过程（对所有样本）就是梯度下降训练。`,
    codeLine: 13,
    activations: { a0, z1: z1new, a1: a1new, z2: z2new, a2: a2new },
    gradients: null,
    loss: lossNew,
    lossHistory: [loss, lossNew],
    highlightNodes: [],
    highlightEdges: [],
  });

  return steps;
}

// Export for use in main.js
window.buildSteps = buildSteps;
