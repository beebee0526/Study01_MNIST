/**
 * UI 로직: 캔버스 그리기 → 전처리(MnistPreprocess) → 추론(MnistNN) → 결과 표시
 */
(function () {
  'use strict';

  const BRUSH = 18; // desktop_version/draw_app.py 와 동일
  const $ = (id) => document.getElementById(id);

  const canvas = $('draw-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const preview = $('preview-canvas');
  const pctx = preview.getContext('2d');
  const bars = $('prob-bars');
  const predDigit = $('pred-digit');
  const predConf = $('pred-conf');

  let model = null;
  let drawing = false;
  let last = null;
  let pending = false;

  // ---------- 모델 로드 ----------
  function loadModel() {
    const badge = $('model-badge');
    try {
      model = MnistNN.prepareModel(window.MNIST_MODEL);
    } catch (err) {
      badge.textContent = '모델 로드 실패';
      badge.className = 'badge error';
      $('model-meta').textContent = String(err.message || err);
      return;
    }
    const m = model.meta;
    const bootstrap = String(m.source || '').includes('bootstrap');
    badge.textContent = bootstrap ? '임시 모델 (합성 데이터 학습)' : 'MNIST 학습 모델';
    badge.className = 'badge ' + (bootstrap ? 'warn' : 'ok');
    const parts = [];
    if (m.test_accuracy != null) parts.push(`MNIST 테스트 정확도 ${(m.test_accuracy * 100).toFixed(2)}%`);
    if (m.sklearn_digits_accuracy != null) parts.push(`실제 손글씨 참고 정확도 ${(m.sklearn_digits_accuracy * 100).toFixed(1)}% (sklearn digits)`);
    if (bootstrap) parts.push('MNIST 로 학습한 가중치로 교체 권장');
    if (m.created) parts.push(`생성 ${m.created.replace('T', ' ')}`);
    $('model-meta').textContent = parts.join(' · ');
  }

  // ---------- 결과 막대 ----------
  const barEls = [];
  for (let d = 0; d < 10; d++) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="digit">${d}</span><span class="track"><span class="fill"></span></span><span class="pct">0.0%</span>`;
    bars.appendChild(li);
    barEls.push({ li, fill: li.querySelector('.fill'), pct: li.querySelector('.pct') });
  }

  function showResult(res) {
    if (!res) {
      predDigit.textContent = '?';
      predConf.textContent = '숫자를 그려 주세요';
      barEls.forEach((b) => { b.fill.style.width = '0'; b.pct.textContent = '0.0%'; b.li.classList.remove('top'); });
      return;
    }
    predDigit.textContent = String(res.digit);
    predConf.textContent = `확신도 ${(res.probs[res.digit] * 100).toFixed(1)}%`;
    res.probs.forEach((p, d) => {
      barEls[d].fill.style.width = (p * 100).toFixed(1) + '%';
      barEls[d].pct.textContent = (p * 100).toFixed(1) + '%';
      barEls[d].li.classList.toggle('top', d === res.digit);
    });
  }

  function drawPreview(x28) {
    const img = pctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = x28 ? Math.round(Math.min(1, Math.max(0, x28[i])) * 255) : 0;
      img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
      img.data[i * 4 + 3] = 255;
    }
    pctx.putImageData(img, 0, 0);
  }

  // ---------- 추론 ----------
  function runInference() {
    pending = false;
    if (!model) return;
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const gray = MnistPreprocess.imageDataToGray(imageData);
    const x28 = MnistPreprocess.centerDigit(gray, canvas.width, canvas.height);
    drawPreview(x28);
    showResult(x28 ? MnistNN.predict(model, x28) : null);
  }

  function schedule() {
    if (!pending) {
      pending = true;
      requestAnimationFrame(runInference);
    }
  }

  // ---------- 그리기 ----------
  function resetCanvas() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    drawPreview(null);
    showResult(null);
  }

  function pos(e) {
    const r = canvas.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * canvas.width) / r.width, y: ((e.clientY - r.top) * canvas.height) / r.height };
  }

  function stroke(a, b) {
    ctx.strokeStyle = '#fff';
    ctx.fillStyle = '#fff';
    ctx.lineWidth = BRUSH;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  canvas.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    drawing = true;
    last = pos(e);
    stroke(last, last);
    schedule();
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    const p = pos(e);
    stroke(last, p);
    last = p;
    schedule();
  });
  const end = () => { if (drawing) { drawing = false; last = null; schedule(); } };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);

  $('clear-btn').addEventListener('click', resetCanvas);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' || e.key === 'Delete') resetCanvas();
  });

  // ---------- 이미지 파일 불러오기 ----------
  $('file-input').addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const W = canvas.width, H = canvas.height;
      const s = Math.min(W / img.width, H / img.height);
      const w = Math.max(1, Math.round(img.width * s));
      const h = Math.max(1, Math.round(img.height * s));
      const off = document.createElement('canvas');
      off.width = w; off.height = h;
      const octx = off.getContext('2d');
      octx.drawImage(img, 0, 0, w, h);
      const gray = MnistPreprocess.imageDataToGray(octx.getImageData(0, 0, w, h)); // 흰 글씨/검은 배경으로 정규화
      resetCanvas();
      const out = ctx.createImageData(w, h);
      for (let i = 0; i < w * h; i++) {
        const v = Math.round(gray[i] * 255);
        out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = v;
        out.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(out, Math.floor((W - w) / 2), Math.floor((H - h) / 2));
      URL.revokeObjectURL(url);
      runInference();
    };
    img.onerror = () => { predConf.textContent = '이미지를 읽을 수 없습니다'; URL.revokeObjectURL(url); };
    img.src = url;
    e.target.value = '';
  });

  loadModel();
  resetCanvas();
})();
