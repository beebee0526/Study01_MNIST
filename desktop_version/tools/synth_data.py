"""합성 손글씨 숫자 생성기 (실제 MNIST 를 받을 수 없을 때의 임시 학습 데이터).

두 가지 소스를 섞는다.
  1) 획(stroke) 템플릿: 숫자별 손글씨 모양 폴리라인을 무작위 변형 후 둥근 펜으로 그림
     → 웹 캔버스에서 마우스로 그린 글씨와 비슷한 분포
  2) 시스템 글꼴: 다양한 글꼴로 숫자를 렌더링 + 두께/기울기/탄성 변형
모든 샘플은 preprocess.center_digit 로 MNIST 형태(28x28)로 변환된다.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocess import center_digit  # noqa: E402

R = 128  # 렌더링 해상도


def arc(cx, cy, rx, ry, a0, a1, n=None):
    n = n or max(8, int(abs(a1 - a0) / 10))
    t = np.radians(np.linspace(a0, a1, n))
    return list(zip(cx + rx * np.cos(t), cy + ry * np.sin(t)))


def L(*pts):
    return list(pts)


# 좌표계: x 0(왼쪽)~1, y 0(위)~1(아래). 숫자별 여러 변형(획 목록의 목록)
TEMPLATES: dict[int, list[list[list[tuple]]]] = {
    0: [[arc(0.5, 0.5, 0.3, 0.46, -90, 270)],
        [arc(0.5, 0.5, 0.26, 0.47, -100, 275)]],
    1: [[L((0.5, 0.0), (0.5, 1.0))],
        [L((0.3, 0.2), (0.52, 0.0), (0.52, 1.0))],
        [L((0.3, 0.2), (0.52, 0.0), (0.52, 1.0)), L((0.3, 1.0), (0.74, 1.0))]],
    2: [[arc(0.5, 0.3, 0.27, 0.27, 200, 380) + L((0.2, 1.0), (0.84, 1.0))],
        [arc(0.48, 0.28, 0.28, 0.28, 190, 350) + L((0.72, 0.5), (0.18, 1.0), (0.85, 0.98))],
        [arc(0.5, 0.32, 0.3, 0.32, 180, 390) + L((0.25, 0.95), (0.18, 1.0), (0.84, 0.96))]],
    3: [[arc(0.48, 0.26, 0.26, 0.24, 200, 450) + arc(0.48, 0.74, 0.29, 0.26, 270, 520)],
        [L((0.2, 0.0), (0.78, 0.0), (0.45, 0.42)) + arc(0.48, 0.72, 0.3, 0.28, 280, 520)]],
    4: [[L((0.6, 0.0), (0.12, 0.66), (0.88, 0.66)), L((0.66, 0.28), (0.66, 1.0))],
        [L((0.22, 0.0), (0.18, 0.6), (0.85, 0.6)), L((0.68, 0.0), (0.68, 1.0))],
        [L((0.66, 1.0), (0.66, 0.0), (0.12, 0.68), (0.9, 0.68))]],
    5: [[L((0.78, 0.0), (0.3, 0.0), (0.26, 0.46)) + arc(0.48, 0.7, 0.29, 0.29, 225, 510)],
        [L((0.3, 0.0), (0.26, 0.46)) + arc(0.48, 0.7, 0.29, 0.29, 225, 510), L((0.3, 0.0), (0.8, 0.0))]],
    6: [[L((0.7, 0.0), (0.46, 0.14), (0.3, 0.36), (0.23, 0.64)) + arc(0.5, 0.73, 0.27, 0.26, 180, 540)],
        [arc(0.7, 0.62, 0.48, 0.62, 250, 180, 10) + arc(0.5, 0.74, 0.28, 0.25, 180, 530)]],
    7: [[L((0.16, 0.0), (0.84, 0.0), (0.42, 1.0))],
        [L((0.16, 0.0), (0.84, 0.0), (0.42, 1.0)), L((0.36, 0.5), (0.78, 0.5))],
        [L((0.16, 0.14), (0.16, 0.0), (0.84, 0.0), (0.5, 1.0))]],
    8: [[arc(0.5, 0.25, 0.22, 0.24, 90, 450), arc(0.5, 0.74, 0.28, 0.26, 270, 630)],
        [arc(0.5, 0.24, 0.2, 0.23, 90, 450), arc(0.5, 0.72, 0.25, 0.28, 270, 630)]],
    9: [[arc(0.5, 0.3, 0.27, 0.29, 0, -360) + L((0.77, 0.3), (0.75, 1.0))],
        [arc(0.5, 0.3, 0.27, 0.29, 0, -360) + L((0.77, 0.3), (0.72, 0.78), (0.55, 1.0))],
        [arc(0.48, 0.28, 0.27, 0.27, 20, -340) + L((0.74, 0.37), (0.4, 1.0))]],
}


def _affine(rng):
    ang = math.radians(rng.normal(0, 8))
    shear = rng.normal(0, 0.18)
    sx = rng.uniform(0.7, 1.15)
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, -s], [s, c]]) @ np.array([[sx, shear], [0, 1]])


def render_template(digit: int, rng) -> np.ndarray:
    strokes = TEMPLATES[digit][rng.integers(len(TEMPLATES[digit]))]
    A = _affine(rng)
    wx, wy = rng.normal(0, 0.03, 2)
    fx, fy, px, py = rng.uniform(0.5, 1.5, 2).tolist() + rng.uniform(0, 6.28, 2).tolist()
    img = Image.new("L", (R, R), 0)
    d = ImageDraw.Draw(img)
    height = rng.uniform(70, 100)
    width = max(2, int(height * rng.uniform(0.09, 0.2)))
    for st in strokes:
        pts = np.array(st, dtype=np.float64)
        pts = _densify(pts)  # 긴 직선을 잘게 나눠 흔들림 적용
        pts = pts + rng.normal(0, 0.008, pts.shape)
        pts[:, 0] += wx * np.sin(2 * np.pi * fx * pts[:, 1] + px)
        pts[:, 1] += wy * np.sin(2 * np.pi * fy * pts[:, 0] + py)
        pts = (pts - 0.5) @ A.T * height + R / 2
        xy = [tuple(p) for p in pts]
        d.line(xy, fill=255, width=width, joint="curve")
        r = width / 2
        for x, y in (xy[0], xy[-1]):
            d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0


def _densify(pts: np.ndarray, step: float = 0.05) -> np.ndarray:
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(0, 1, n + 1)[1:]:
            out.append(a + (b - a) * t)
    return np.array(out)


def find_fonts(limit: int = 200) -> list[str]:
    """숫자를 제대로 그리는 시스템 글꼴 목록 (기호/두부 글꼴 제외)."""
    try:
        out = subprocess.run(["fc-list", "--format", "%{file}\n"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return []
    files = sorted({f for f in out.splitlines() if f.lower().endswith((".ttf", ".otf"))})
    good = []
    for f in files:
        try:
            font = ImageFont.truetype(f, 64)
            imgs = []
            for ch in "0123456789":
                im = Image.new("L", (R, R), 0)
                ImageDraw.Draw(im).text((R / 2, R / 2), ch, fill=255, font=font, anchor="mm")
                a = center_digit(np.asarray(im, dtype=np.float32) / 255.0)
                if a is None:
                    raise ValueError
                imgs.append(a)
            # 서로 다른 숫자가 충분히 달라야 함 (두부 상자 글꼴 제외)
            diffs = [np.abs(imgs[i] - imgs[j]).mean() for i in range(10) for j in range(i + 1, 10)]
            if min(diffs) > 0.04:
                good.append(f)
        except Exception:
            continue
        if len(good) >= limit:
            break
    return good


def render_font(digit: int, font_path: str, rng) -> np.ndarray:
    font = ImageFont.truetype(font_path, int(rng.uniform(56, 80)))
    img = Image.new("L", (R, R), 0)
    ImageDraw.Draw(img).text((R / 2, R / 2), str(digit), fill=255, font=font, anchor="mm")
    k = rng.choice([0, 0, 3, 3, 5])
    if k:
        img = img.filter(ImageFilter.MaxFilter(int(k)))
    A = _affine(rng)
    inv = np.linalg.inv(A)
    c = np.array([R / 2, R / 2])
    off = c - inv @ c
    img = img.transform((R, R), Image.AFFINE, (inv[0, 0], inv[0, 1], off[0], inv[1, 0], inv[1, 1], off[1]),
                        resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def elastic(img: np.ndarray, rng, alpha=8.0, sigma=5.0) -> np.ndarray:
    dx = gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, img.shape), sigma) * alpha
    y, x = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing="ij")
    return map_coordinates(img, [y + dy, x + dx], order=1, mode="constant")


def make_sample(digit: int, fonts: list[str], rng, template_ratio: float = 0.6) -> np.ndarray:
    for _ in range(5):
        if not fonts or rng.random() < template_ratio:
            raw = render_template(digit, rng)
        else:
            raw = render_font(digit, fonts[rng.integers(len(fonts))], rng)
        if rng.random() < 0.5:
            raw = elastic(raw, rng)
        out = center_digit(raw)
        if out is not None:
            return out
    raise RuntimeError("샘플 생성 실패")


def make_dataset(n: int, fonts: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 10
    rng.shuffle(y)
    X = np.stack([make_sample(int(d), fonts, rng) for d in y])
    return X.astype(np.float32), y.astype(np.int64)
