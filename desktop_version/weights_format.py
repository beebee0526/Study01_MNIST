"""웹용 가중치 파일 포맷 (mnist-cnn-v1) 작성/읽기. PyTorch 불필요.

web_version/model/weights.js 는 다음 형태의 '클래식 스크립트'다.
(ES 모듈/fetch 를 쓰지 않으므로 file:// 로 열어도, GitHub Pages 에서도 동작)

    window.MNIST_MODEL = { "format": "mnist-cnn-v1", "meta": {...},
                           "input": {"size": 28, "mean": 0.1307, "std": 0.3081},
                           "layers": [ {"type": "conv2d", ...}, {"type": "relu"}, ... ] };

레이어 타입과 가중치 배치(PyTorch 와 동일, row-major 평탄화)
  conv2d   : inC, outC, k, weight[outC*inC*k*k] (O,C,kh,kw), bias[outC]   (stride 1, padding 0)
  relu     : -
  maxpool2d: k (stride = k)
  flatten  : - (C,H,W 순서)
  linear   : in, out, weight[out*in] (out,in), bias[out]
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

FORMAT = "mnist-cnn-v1"
MEAN, STD = 0.1307, 0.3081
PREFIX = "window.MNIST_MODEL = "

DEFAULT_WEB_PATH = Path(__file__).resolve().parent.parent / "web_version" / "model" / "weights.js"


def _flat(a, digits: int) -> list[float]:
    return [round(float(v), digits) for v in np.asarray(a, dtype=np.float64).ravel()]


def build_model_dict(layers: list[dict], meta: dict, digits: int = 6) -> dict:
    """layers: {'type': ..., 'weight': ndarray, 'bias': ndarray, ...} 목록 → 직렬화용 dict."""
    out = []
    for layer in layers:
        t = layer["type"]
        if t == "conv2d":
            w = np.asarray(layer["weight"])
            assert w.ndim == 4 and w.shape[2] == w.shape[3], w.shape
            out.append({"type": t, "inC": int(w.shape[1]), "outC": int(w.shape[0]), "k": int(w.shape[2]),
                        "weight": _flat(w, digits), "bias": _flat(layer["bias"], digits)})
        elif t == "linear":
            w = np.asarray(layer["weight"])
            assert w.ndim == 2, w.shape
            out.append({"type": t, "in": int(w.shape[1]), "out": int(w.shape[0]),
                        "weight": _flat(w, digits), "bias": _flat(layer["bias"], digits)})
        elif t == "maxpool2d":
            out.append({"type": t, "k": int(layer.get("k", 2))})
        elif t in ("relu", "flatten"):
            out.append({"type": t})
        else:
            raise ValueError(f"지원하지 않는 레이어: {t}")
    return {"format": FORMAT, "meta": meta,
            "input": {"size": 28, "mean": MEAN, "std": STD}, "layers": out}


def write_weights_js(model: dict, path: Path | str = DEFAULT_WEB_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"/* 자동 생성 파일: desktop_version 에서 생성. 직접 수정하지 마세요. */\n{PREFIX}{body};\n",
                    encoding="utf-8")
    return path


def read_weights_js(path: Path | str = DEFAULT_WEB_PATH) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(re.escape(PREFIX) + r"(\{.*\});?\s*$", text, re.S)
    if not m:
        raise ValueError(f"{path}: MNIST_MODEL 정의를 찾을 수 없음")
    model = json.loads(m.group(1))
    if model.get("format") != FORMAT:
        raise ValueError(f"알 수 없는 포맷: {model.get('format')}")
    return model


def numpy_forward(model: dict, x28: np.ndarray) -> np.ndarray:
    """직렬화된 모델로 NumPy 순전파(참조 구현). x28: [0,1] (28,28) → logits(10)."""
    inp = model["input"]
    x = ((np.asarray(x28, dtype=np.float64) - inp["mean"]) / inp["std"])[None]  # (1,28,28)
    for L in model["layers"]:
        t = L["type"]
        if t == "conv2d":
            k, C, O = L["k"], L["inC"], L["outC"]
            w = np.array(L["weight"]).reshape(O, C, k, k)
            win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))  # C,OH,OW,k,k
            x = np.einsum("chwij,ocij->ohw", win, w) + np.array(L["bias"])[:, None, None]
        elif t == "relu":
            x = np.maximum(x, 0)
        elif t == "maxpool2d":
            k = L["k"]
            if x.ndim == 3:
                C, H, W = x.shape
                x = x[:, : H // k * k, : W // k * k].reshape(C, H // k, k, W // k, k).max(axis=(2, 4))
        elif t == "flatten":
            x = x.ravel()
        elif t == "linear":
            w = np.array(L["weight"]).reshape(L["out"], L["in"])
            x = w @ x + np.array(L["bias"])
    return x
