#!/usr/bin/env node
/**
 * 웹 추론 엔진 테스트 (Node 18+, 외부 패키지 없음)
 *   node tests/run_tests.js
 *
 * 1) 단위 테스트: 면적평균 리사이즈, 중앙 정렬, conv/pool/linear 수작업 모델
 * 2) 일치 테스트: desktop_version/tools/make_js_fixture.py 가 만든 fixture.json 과
 *    전처리 결과·logits 가 Python 참조 구현과 일치하는지 확인
 * 3) index.html 필수 요소(학번/이름 최상단) 확인
 */
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const ROOT = path.resolve(__dirname, '..');
const P = require(path.join(ROOT, 'js/preprocess.js'));
const NN = require(path.join(ROOT, 'js/nn.js'));

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log('  ✓ ' + name); }
  catch (e) { failed++; console.log('  ✗ ' + name + '\n    ' + (e.stack || e).toString().split('\n').slice(0, 3).join('\n    ')); }
}
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b} (tol ${tol})`);

function loadWeights() {
  const ctx = { window: {} };
  vm.runInNewContext(fs.readFileSync(path.join(ROOT, 'model/weights.js'), 'utf8'), ctx);
  return ctx.window.MNIST_MODEL;
}

console.log('전처리');
test('areaResize: 4x4 → 2x2 는 2x2 블록 평균', () => {
  const img = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const out = P.areaResize(img, 4, 4, 2, 2);
  [1, 0, 0, 0.5].forEach((v, i) => close(out[i], v, 1e-12, 'cell ' + i));
});
test('areaResize: 가중치 행 합 = 1 (비정수 배율)', () => {
  const m = P.areaMatrix(7, 3);
  for (let i = 0; i < 3; i++) { let s = 0; for (let j = 0; j < 7; j++) s += m[i * 7 + j]; close(s, 1, 1e-12, 'row ' + i); }
});
test('centerDigit: 빈 이미지는 null', () => {
  assert.strictEqual(P.centerDigit(new Float64Array(100), 10, 10), null);
});
test('centerDigit: 구석의 사각형 → 긴 변 20px, 무게중심 ≈ 13.5', () => {
  const W = 50, g = new Float64Array(W * W);
  for (let y = 0; y < 10; y++) for (let x = 0; x < 5; x++) g[y * W + x] = 1;
  const out = P.centerDigit(g, W, W);
  let tot = 0, sy = 0, sx = 0, rows = new Set();
  for (let i = 0; i < 784; i++) if (out[i] > 0) { tot += out[i]; sy += out[i] * Math.floor(i / 28); sx += out[i] * (i % 28); rows.add(Math.floor(i / 28)); }
  assert.strictEqual(rows.size, 20);
  close(sy / tot, 13.5, 0.5, 'cy'); close(sx / tot, 13.5, 0.5, 'cx');
});
test('imageDataToGray: 흰 바탕/검은 글씨는 반전', () => {
  const data = new Uint8ClampedArray(4 * 9).fill(255);
  data.set([0, 0, 0, 255], 4 * 4); // 가운데 검은 점
  const g = P.imageDataToGray({ data, width: 3, height: 3 });
  close(g[4], 1, 1e-9, 'center'); close(g[0], 0, 1e-9, 'corner');
});

console.log('추론 엔진');
test('수작업 모델: conv(합) → relu → pool → flatten → linear', () => {
  const k = 27; // 28-27+1 = 2x2 출력
  const raw = {
    format: 'mnist-cnn-v1', input: { size: 28, mean: 0, std: 1 },
    layers: [
      { type: 'conv2d', inC: 1, outC: 1, k, weight: new Array(k * k).fill(1), bias: [-1] },
      { type: 'relu' }, { type: 'maxpool2d', k: 2 }, { type: 'flatten' },
      { type: 'linear', in: 1, out: 2, weight: [1, -1], bias: [0, 0.5] },
    ],
  };
  const x = new Float32Array(784); x[0] = 3; // 좌상단 창에만 포함
  const logits = NN.forward(NN.prepareModel(raw), x);
  close(logits[0], 2, 1e-6, 'l0'); close(logits[1], -1.5, 1e-6, 'l1');
});
test('softmax 합 = 1', () => {
  const p = NN.softmax([1, 2, 3, 1000]);
  close(p.reduce((a, b) => a + b, 0), 1, 1e-12, 'sum');
});
test('잘못된 포맷은 예외', () => {
  assert.throws(() => NN.prepareModel({ format: 'x', layers: [] }));
});

console.log('weights.js ↔ Python 참조 구현 일치');
const weights = loadWeights();
test('weights.js 로드 및 구조 확인', () => {
  const m = NN.prepareModel(weights);
  assert.deepStrictEqual(Array.from(m.layers, (l) => l.type),
    ['conv2d', 'relu', 'maxpool2d', 'conv2d', 'relu', 'maxpool2d', 'flatten', 'linear', 'relu', 'linear']);
});
const fixturePath = path.join(__dirname, 'fixture.json');
if (!fs.existsSync(fixturePath)) {
  failed++; console.log('  ✗ fixture.json 없음: python desktop_version/tools/make_js_fixture.py 실행 필요');
} else {
  const fx = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const model = NN.prepareModel(weights);
  test('fixture 가 현재 weights.js 로 생성됨', () => assert.strictEqual(fx.source, weights.meta.source));
  fx.cases.forEach((c, idx) => {
    test(`case ${idx} (${c.w}x${c.h}) 전처리·logits 일치, 예측 ${c.argmax}`, () => {
      const x28 = P.centerDigit(c.raw, c.w, c.h);
      let maxDiff = 0;
      for (let i = 0; i < 784; i++) maxDiff = Math.max(maxDiff, Math.abs(x28[i] - c.x28[i]));
      assert.ok(maxDiff < 1e-5, 'x28 최대 오차 ' + maxDiff);
      const logits = NN.forward(model, x28);
      c.logits.forEach((v, i) => close(logits[i], v, 1e-3, 'logit ' + i));
      assert.strictEqual(NN.predict(model, x28).digit, c.argmax);
    });
  });
}

console.log('index.html');
test('학번/이름이 body 의 첫 요소로 노출', () => {
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  const body = html.split(/<body[^>]*>/i)[1].replace(/<!--[\s\S]*?-->/g, '').trim();
  assert.ok(/^<header[^>]*>학번: 2601949 \| 이름: 송은비<\/header>/.test(body), body.slice(0, 120));
});
test('외부 스크립트/스타일 URL 미사용', () => {
  const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  assert.ok(!/(src|href)="(https?:)?\/\//i.test(html));
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
