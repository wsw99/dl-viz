/**
 * controls.js — Shared step engine for DL Visualizer
 * Usage: StepEngine.init({ steps, renderFn, totalEl, counterEl })
 */
const StepEngine = (() => {
  let _steps      = [];
  let _index      = 0;
  let _renderFn   = null;
  let _timer      = null;
  let _speed      = 1200; // ms per auto-play step

  // DOM refs (set in init)
  let btnPrev, btnNext, btnReset, btnPlay, counterEl;

  function init({ steps, renderFn }) {
    _steps   = steps;
    _index   = 0;
    _renderFn = renderFn;

    btnPrev   = document.getElementById('btn-prev');
    btnNext   = document.getElementById('btn-next');
    btnReset  = document.getElementById('btn-reset');
    btnPlay   = document.getElementById('btn-play');
    counterEl = document.getElementById('step-counter');

    const speedSel = document.getElementById('speed-select');
    if (speedSel) {
      speedSel.addEventListener('change', () => {
        _speed = parseInt(speedSel.value, 10);
        if (_timer) { stopPlay(); startPlay(); } // restart with new speed
      });
    }

    btnPrev.addEventListener('click',  prevStep);
    btnNext.addEventListener('click',  nextStep);
    btnReset.addEventListener('click', reset);
    btnPlay.addEventListener('click',  togglePlay);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') nextStep();
      if (e.key === 'ArrowLeft')  prevStep();
      if (e.key === 'r' || e.key === 'R') reset();
      if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    });

    _render();
  }

  function _render() {
    if (!_renderFn || _steps.length === 0) return;
    _renderFn(_steps[_index], _index, _steps.length);
    if (counterEl) counterEl.textContent = `Step ${_index} / ${_steps.length - 1}`;
    if (btnPrev)  btnPrev.disabled  = _index === 0;
    if (btnNext)  btnNext.disabled  = _index === _steps.length - 1;
  }

  function nextStep() {
    if (_index < _steps.length - 1) { _index++; _render(); }
    else stopPlay();
  }

  function prevStep() {
    if (_index > 0) { _index--; _render(); }
  }

  function reset() {
    stopPlay();
    _index = 0;
    _render();
  }

  function startPlay() {
    if (_timer) return;
    btnPlay.textContent = '⏸ Pause';
    _timer = setInterval(() => {
      if (_index < _steps.length - 1) nextStep();
      else stopPlay();
    }, _speed);
  }

  function stopPlay() {
    if (_timer) { clearInterval(_timer); _timer = null; }
    if (btnPlay) btnPlay.textContent = '⏩ Auto Play';
  }

  function togglePlay() {
    _timer ? stopPlay() : startPlay();
  }

  // Allow external jump (e.g. click a node)
  function jumpTo(i) {
    stopPlay();
    _index = Math.max(0, Math.min(i, _steps.length - 1));
    _render();
  }

  return { init, jumpTo };
})();
