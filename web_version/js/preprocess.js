/**
 * MNIST 스타일 전처리 (순수 JS, 외부 라이브러리 없음)
 *
 * desktop_version/preprocess.py 와 "완전히 같은" 알고리즘이다. 한쪽을 바꾸면 다른 쪽도 바꾸고
 * desktop_version/tools/make_js_fixture.py 로 테스트 fixture 를 다시 만든다.
 *
 *  1) 흑백 [0,1] 배열, 글씨 = 밝은 값. 테두리가 밝으면 반전
 *  2) INK_THRESHOLD 초과 픽셀의 경계상자로 자르기
 *  3) 긴 변 20px, 종횡비 유지, 면적평균 리샘플링
 *  4) 28x28 중앙 배치 후 무게중심이 (13.5, 13.5)에 오도록 정수 이동
 *
 * 브라우저: window.MnistPreprocess / Node: require('./preprocess.js')
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.MnistPreprocess = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const SIZE = 28;
  const BOX = 20;
  const INK_THRESHOLD = 0.1;
  const CENTER = (SIZE - 1) / 2;

  /** Python 쪽 round_half_up 과 동일 */
  const roundHalfUp = (x) => Math.floor(x + 0.5);

  /** src 길이 → dst 길이 면적평균 가중치 행렬 (dst x src, row-major) */
  function areaMatrix(src, dst) {
    const m = new Float64Array(dst * src);
    const scale = src / dst;
    for (let i = 0; i < dst; i++) {
      const a = i * scale;
      const b = (i + 1) * scale;
      const j0 = Math.floor(a);
      const j1 = Math.min(Math.ceil(b), src);
      for (let j = j0; j < j1; j++) {
        const overlap = Math.min(b, j + 1) - Math.max(a, j);
        if (overlap > 0) m[i * src + j] = overlap / scale;
      }
    }
    return m;
  }

  /** (h x w) → (nh x nw) 면적평균 리사이즈. 반환: Float64Array(nh*nw) */
  function areaResize(img, h, w, nh, nw) {
    const ry = areaMatrix(h, nh);
    const rx = areaMatrix(w, nw);
    const tmp = new Float64Array(nh * w); // ry @ img
    for (let i = 0; i < nh; i++) {
      for (let j = 0; j < h; j++) {
        const r = ry[i * h + j];
        if (r === 0) continue;
        for (let x = 0; x < w; x++) tmp[i * w + x] += r * img[j * w + x];
      }
    }
    const out = new Float64Array(nh * nw); // tmp @ rx^T
    for (let i = 0; i < nh; i++) {
      for (let x = 0; x < nw; x++) {
        let s = 0;
        for (let j = 0; j < w; j++) s += tmp[i * w + j] * rx[x * w + j];
        out[i * nw + x] = s;
      }
    }
    return out;
  }

  const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

  /**
   * 흰 글씨/검은 배경 [0,1] 배열(width x height) → Float32Array(784), 글씨가 없으면 null
   */
  function centerDigit(gray, width, height) {
    let y0 = height, y1 = -1, x0 = width, x1 = -1;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (gray[y * width + x] > INK_THRESHOLD) {
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
        }
      }
    }
    if (y1 < 0) return null;

    const h = y1 - y0 + 1;
    const w = x1 - x0 + 1;
    const crop = new Float64Array(h * w);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) crop[y * w + x] = gray[(y0 + y) * width + (x0 + x)];
    }
    const s = BOX / Math.max(h, w);
    const nh = Math.max(1, roundHalfUp(h * s));
    const nw = Math.max(1, roundHalfUp(w * s));
    const small = areaResize(crop, h, w, nh, nw);

    const top = Math.floor((SIZE - nh) / 2);
    const left = Math.floor((SIZE - nw) / 2);
    let total = 0, sy = 0, sx = 0;
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) {
        const v = small[y * nw + x];
        total += v;
        sy += v * (top + y);
        sx += v * (left + x);
      }
    }
    if (total <= 0) return null;
    let dy = roundHalfUp(CENTER - sy / total);
    let dx = roundHalfUp(CENTER - sx / total);
    dy = clamp(dy, -top, SIZE - (top + nh));
    dx = clamp(dx, -left, SIZE - (left + nw));

    const out = new Float32Array(SIZE * SIZE);
    for (let y = 0; y < nh; y++) {
      for (let x = 0; x < nw; x++) out[(top + dy + y) * SIZE + (left + dx + x)] = small[y * nw + x];
    }
    return out;
  }

  /**
   * ImageData(RGBA) → [0,1] 흑백 Float64Array. 투명 픽셀은 흰 배경으로 합성,
   * 테두리가 밝으면(흰 바탕/검은 글씨) 반전하여 항상 흰 글씨/검은 배경으로 만든다.
   */
  function imageDataToGray(imageData) {
    const { data, width, height } = imageData;
    const gray = new Float64Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const a = data[i * 4 + 3] / 255;
      const lum = (data[i * 4] + data[i * 4 + 1] + data[i * 4 + 2]) / (3 * 255);
      gray[i] = lum * a + 1 * (1 - a);
    }
    let sum = 0, n = 0;
    for (let x = 0; x < width; x++) { sum += gray[x] + gray[(height - 1) * width + x]; n += 2; }
    for (let y = 0; y < height; y++) { sum += gray[y * width] + gray[y * width + width - 1]; n += 2; }
    if (sum / n > 0.5) for (let i = 0; i < gray.length; i++) gray[i] = 1 - gray[i];
    return gray;
  }

  return { SIZE, BOX, INK_THRESHOLD, roundHalfUp, areaMatrix, areaResize, centerDigit, imageDataToGray };
});
