"""MNIST 스타일 전처리 (NumPy 전용, PyTorch 불필요).

web_version/js/preprocess.js 와 **완전히 같은 알고리즘**을 구현한다.
한쪽을 바꾸면 반드시 다른 쪽도 함께 바꾸고 테스트(fixture)를 다시 만든다.

알고리즘
  1) 흑백 [0,1] 배열, 글씨 = 밝은 값(흰 글씨/검은 배경). 배경이 밝으면 반전.
  2) INK_THRESHOLD 초과 픽셀의 경계상자(bbox)로 자르기
  3) 긴 변이 20px 가 되도록 종횡비 유지 + 면적평균(area) 리샘플링
  4) 28x28 중앙에 놓은 뒤, 무게중심이 (13.5, 13.5)에 오도록 정수 이동
"""
from __future__ import annotations

import math

import numpy as np

SIZE = 28
BOX = 20
INK_THRESHOLD = 0.1
CENTER = (SIZE - 1) / 2.0  # 13.5


def round_half_up(x: float) -> int:
    """JS Math.round 와 동일한 반올림 (Python round 는 banker's rounding)."""
    return int(math.floor(x + 0.5))


def _area_matrix(src: int, dst: int) -> np.ndarray:
    """src 길이를 dst 길이로 면적평균 리샘플링하는 (dst, src) 가중치 행렬."""
    m = np.zeros((dst, src), dtype=np.float64)
    scale = src / dst
    for i in range(dst):
        a, b = i * scale, (i + 1) * scale
        j0, j1 = int(math.floor(a)), min(int(math.ceil(b)), src)
        for j in range(j0, j1):
            overlap = min(b, j + 1) - max(a, j)
            if overlap > 0:
                m[i, j] = overlap / scale
    return m


def area_resize(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """분리 가능한 면적평균 리사이즈. img: (H, W) float."""
    ry = _area_matrix(img.shape[0], out_h)
    rx = _area_matrix(img.shape[1], out_w)
    return ry @ img.astype(np.float64) @ rx.T


def center_digit(gray: np.ndarray) -> np.ndarray | None:
    """흰 글씨/검은 배경 [0,1] 배열 → 28x28 float32. 글씨가 없으면 None."""
    gray = np.asarray(gray, dtype=np.float64)
    ys, xs = np.nonzero(gray > INK_THRESHOLD)
    if ys.size == 0:
        return None
    crop = gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    s = BOX / max(h, w)
    nh, nw = max(1, round_half_up(h * s)), max(1, round_half_up(w * s))
    small = area_resize(crop, nh, nw)

    out = np.zeros((SIZE, SIZE), dtype=np.float64)
    top, left = (SIZE - nh) // 2, (SIZE - nw) // 2
    out[top: top + nh, left: left + nw] = small

    total = out.sum()
    if total <= 0:
        return None
    rows, cols = np.arange(SIZE), np.arange(SIZE)
    cy = (out.sum(axis=1) * rows).sum() / total
    cx = (out.sum(axis=0) * cols).sum() / total
    dy, dx = round_half_up(CENTER - cy), round_half_up(CENTER - cx)
    # 글씨가 잘리지 않도록 이동량 제한
    dy = int(np.clip(dy, -top, SIZE - (top + nh)))
    dx = int(np.clip(dx, -left, SIZE - (left + nw)))
    shifted = np.zeros_like(out)
    shifted[top + dy: top + dy + nh, left + dx: left + dx + nw] = small
    return shifted.astype(np.float32)


def to_gray(image) -> np.ndarray:
    """PIL.Image 또는 배열 → [0,1] 흑백, 흰 글씨/검은 배경으로 정규화."""
    if hasattr(image, "convert"):  # PIL.Image
        if image.mode in ("RGBA", "LA"):
            from PIL import Image
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            bg.alpha_composite(image.convert("RGBA"))
            image = bg
        arr = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    else:
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        if arr.max() > 1.0:
            arr = arr / 255.0
    # 테두리 평균이 밝으면 흰 바탕/검은 글씨로 보고 반전
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if border.mean() > 0.5:
        arr = 1.0 - arr
    return arr


def image_to_mnist(image) -> np.ndarray | None:
    """임의 이미지 → 28x28 float32 [0,1] (정규화 전). 글씨가 없으면 None."""
    return center_digit(to_gray(image))
