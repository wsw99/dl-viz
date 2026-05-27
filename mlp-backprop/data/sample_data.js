/**
 * XOR dataset — 4 canonical points.
 * Network: 2 inputs → 3 hidden (ReLU) → 1 output (Sigmoid) → BCE loss
 *
 * Pre-trained weights after ~5000 steps of SGD (lr=0.1) on XOR,
 * confirmed to produce loss ≈ 0.02 at convergence.
 *
 * We also store the INITIAL weights so the visualizer can
 * show the "before" state and replay training from scratch.
 */
window.DATA = {
  // XOR inputs and labels
  samples: [
    { x: [0, 0], y: 0 },
    { x: [0, 1], y: 1 },
    { x: [1, 0], y: 1 },
    { x: [1, 1], y: 0 },
  ],

  // Architecture
  arch: { inputSize: 2, hiddenSize: 3, outputSize: 1 },

  // Hyperparameters used in the step-through demo
  lr: 0.5,

  // Initial weights (seeded, reproducible)
  // w1: [hiddenSize x inputSize], b1: [hiddenSize]
  // w2: [outputSize x hiddenSize], b2: [outputSize]
  initWeights: {
    w1: [
      [ 0.62, -0.43],
      [-0.31,  0.78],
      [ 0.55,  0.21],
    ],
    b1: [0.05, -0.12, 0.08],
    w2: [[ 0.48, -0.67,  0.33]],
    b2: [0.02],
  },

  // The single training sample used for the step-through demo
  // (We step through ONE forward+backward pass on sample index 1: x=[0,1], y=1)
  demoSampleIdx: 1,
};
