"""Python(참조) ↔ JS 일치 검증용 fixture 생성.

web_version/model/weights.js 를 읽어 NumPy 참조 구현으로
  (1) 원본 이미지 → 28x28 전처리 결과, (2) logits 를 계산하고
web_version/tests/fixture.json 에 저장한다. JS 테스트(node tests/run_tests.js)가 이를 비교한다.

가중치나 전처리를 바꾼 뒤에는 반드시 다시 실행:
    python tools/make_js_fixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from preprocess import center_digit  # noqa: E402
from weights_format import DEFAULT_WEB_PATH, numpy_forward, read_weights_js  # noqa: E402

OUT = HERE.parent.parent / "web_version" / "tests" / "fixture.json"


def raw_cases(rng) -> list[np.ndarray]:
    from synth_data import render_template
    cases = []
    for d in range(10):  # 합성 손글씨를 다양한 크기로 축소 (종횡비/크기 경계 조건 포함)
        img = render_template(d, rng)
        h, w = int(rng.integers(30, 100)), int(rng.integers(30, 100))
        from PIL import Image
        img = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
        cases.append(img)
    thin = np.zeros((40, 40)); thin[5:35, 20] = 1.0  # 1픽셀 세로선 (극단적 종횡비)
    dot = np.zeros((30, 30)); dot[14, 14] = 0.8      # 점 하나
    corner = np.zeros((50, 50)); corner[:8, :8] = 1.0  # 구석 글씨
    return cases + [thin, dot, corner]


def main():
    model = read_weights_js(DEFAULT_WEB_PATH)
    rng = np.random.default_rng(1234)
    cases = []
    for raw in raw_cases(rng):
        raw = np.round(raw, 4)
        x28 = center_digit(raw)
        logits = numpy_forward(model, x28)
        cases.append({"h": raw.shape[0], "w": raw.shape[1], "raw": raw.ravel().tolist(),
                      "x28": np.round(x28.astype(np.float64), 7).ravel().tolist(),
                      "logits": np.round(logits, 6).tolist(), "argmax": int(np.argmax(logits))})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": model["meta"].get("source"), "cases": cases}), encoding="utf-8")
    print(f"fixture {len(cases)}건 → {OUT}")


if __name__ == "__main__":
    main()
