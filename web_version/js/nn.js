/**
 * 순수 JS CNN 추론 엔진 (외부 라이브러리 없음)
 *
 * model/weights.js 의 window.MNIST_MODEL (포맷 mnist-cnn-v1) 을 읽어 순전파한다.
 * 지원 레이어: conv2d(stride 1, padding 0), relu, maxpool2d(stride=k), flatten, linear
 * 가중치 배치는 PyTorch 와 동일: conv (O,C,k,k), linear (out,in) 을 row-major 로 평탄화.
 *
 * 브라우저: window.MnistNN / Node: require('./nn.js')
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.MnistNN = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const FORMAT = 'mnist-cnn-v1';

  /** JSON 모델을 검증하고 가중치를 Float32Array 로 변환 */
  function prepareModel(raw) {
    if (!raw || raw.format !== FORMAT) throw new Error('지원하지 않는 모델 포맷: ' + (raw && raw.format));
    const layers = raw.layers.map((L) => {
      const out = Object.assign({}, L);
      if (L.weight) out.weight = Float32Array.from(L.weight);
      if (L.bias) out.bias = Float32Array.from(L.bias);
      if (L.type === 'conv2d' && out.weight.length !== L.outC * L.inC * L.k * L.k) throw new Error('conv2d 가중치 크기 불일치');
      if (L.type === 'linear' && out.weight.length !== L.out * L.in) throw new Error('linear 가중치 크기 불일치');
      return out;
    });
    return { meta: raw.meta || {}, input: raw.input, layers };
  }

  function conv2d(x, shape, L) {
    const [C, H, W] = shape;
    if (C !== L.inC) throw new Error(`conv2d 입력 채널 불일치: ${C} != ${L.inC}`);
    const k = L.k, O = L.outC, OH = H - k + 1, OW = W - k + 1;
    const w = L.weight, b = L.bias;
    const out = new Float32Array(O * OH * OW);
    for (let o = 0; o < O; o++) {
      for (let oy = 0; oy < OH; oy++) {
        for (let ox = 0; ox < OW; ox++) {
          let s = b[o];
          for (let c = 0; c < C; c++) {
            const wBase = (o * C + c) * k * k;
            const xBase = c * H * W;
            for (let ky = 0; ky < k; ky++) {
              const row = xBase + (oy + ky) * W + ox;
              const wr = wBase + ky * k;
              for (let kx = 0; kx < k; kx++) s += w[wr + kx] * x[row + kx];
            }
          }
          out[(o * OH + oy) * OW + ox] = s;
        }
      }
    }
    return [out, [O, OH, OW]];
  }

  function relu(x, shape) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    return [out, shape];
  }

  function maxpool2d(x, shape, L) {
    const [C, H, W] = shape;
    const k = L.k, OH = Math.floor(H / k), OW = Math.floor(W / k);
    const out = new Float32Array(C * OH * OW);
    for (let c = 0; c < C; c++) {
      for (let oy = 0; oy < OH; oy++) {
        for (let ox = 0; ox < OW; ox++) {
          let m = -Infinity;
          for (let ky = 0; ky < k; ky++) {
            for (let kx = 0; kx < k; kx++) {
              const v = x[(c * H + oy * k + ky) * W + ox * k + kx];
              if (v > m) m = v;
            }
          }
          out[(c * OH + oy) * OW + ox] = m;
        }
      }
    }
    return [out, [C, OH, OW]];
  }

  function linear(x, shape, L) {
    if (x.length !== L.in) throw new Error(`linear 입력 크기 불일치: ${x.length} != ${L.in}`);
    const out = new Float32Array(L.out);
    const w = L.weight;
    for (let o = 0; o < L.out; o++) {
      let s = L.bias[o];
      const base = o * L.in;
      for (let i = 0; i < L.in; i++) s += w[base + i] * x[i];
      out[o] = s;
    }
    return [out, [L.out]];
  }

  const OPS = {
    conv2d,
    relu,
    maxpool2d,
    flatten: (x, shape) => [x, [x.length]],
    linear,
  };

  /** x28: 길이 784, [0,1] (정규화 전). 반환: logits Float32Array(10) */
  function forward(model, x28) {
    const { mean, std, size } = model.input;
    let x = new Float32Array(size * size);
    for (let i = 0; i < x.length; i++) x[i] = (x28[i] - mean) / std;
    let shape = [1, size, size];
    for (const L of model.layers) {
      const op = OPS[L.type];
      if (!op) throw new Error('지원하지 않는 레이어: ' + L.type);
      [x, shape] = op(x, shape, L);
    }
    return x;
  }

  function softmax(logits) {
    let m = -Infinity;
    for (const v of logits) if (v > m) m = v;
    const e = Array.from(logits, (v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  function predict(model, x28) {
    const logits = forward(model, x28);
    const probs = softmax(logits);
    let digit = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[digit]) digit = i;
    return { digit, probs, logits };
  }

  return { FORMAT, prepareModel, forward, softmax, predict };
});
