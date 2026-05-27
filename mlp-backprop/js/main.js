/**
 * main.js — Page initialisation for mlp-backprop
 */
document.addEventListener('DOMContentLoaded', () => {
  // 1. Build steps from data
  const steps = buildSteps(window.DATA);

  // 2. Init canvas & edge list
  Viz.initCanvas();
  Viz.buildEdges();
  Viz.initLossChart();

  // 3. Hand over to step engine
  StepEngine.init({
    steps,
    renderFn: (step, idx, total) => Viz.render(step, idx, total),
  });
});
