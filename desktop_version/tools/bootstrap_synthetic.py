"""[임시 모델] PyTorch/MNIST 없이 NumPy + 합성 손글씨로 웹용 가중치를 만든다.

실제 배포 전에는 반드시 `python train.py && python export_weights.py` 로 교체할 것.
이 스크립트는 네트워크가 막힌 개발 환경에서 웹 앱을 바로 시연할 수 있도록 만든 부트스트랩이다.

    python tools/bootstrap_synthetic.py --train 60000 --epochs 6
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from numpy_cnn import MEAN, STD, NumpyCNN  # noqa: E402
from synth_data import find_fonts, make_dataset  # noqa: E402
from weights_format import DEFAULT_WEB_PATH, build_model_dict, write_weights_js  # noqa: E402


def _chunk(args):
    n, fonts, seed = args
    return make_dataset(n, fonts, seed)


def build(n: int, fonts, seed: int, workers: int):
    per = [n // workers + (i < n % workers) for i in range(workers)]
    with Pool(workers) as pool:
        parts = pool.map(_chunk, [(k, fonts, seed * 100 + i) for i, k in enumerate(per)])
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return X, y


def sklearn_digits_eval(model: NumpyCNN) -> float | None:
    """실제 사람 손글씨(sklearn digits, 8x8 → 업샘플) 에 대한 참고용 정확도."""
    try:
        from sklearn.datasets import load_digits
    except ImportError:
        return None
    from PIL import Image
    from preprocess import center_digit
    d = load_digits()
    xs = []
    for im in d.images:
        big = np.asarray(Image.fromarray((im / 16 * 255).astype(np.uint8)).resize((64, 64), Image.BICUBIC),
                         dtype=np.float32) / 255.0
        xs.append(center_digit(big))
    return float((model.predict(np.stack(xs)) == d.target).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=int, default=60000)
    p.add_argument("--val", type=int, default=6000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=DEFAULT_WEB_PATH)
    args = p.parse_args()

    t0 = time.time()
    fonts = find_fonts()
    print(f"글꼴 {len(fonts)}개 사용")
    Xtr, ytr = build(args.train, fonts, args.seed, args.workers)
    Xva, yva = build(args.val, fonts, args.seed + 999, args.workers)
    print(f"데이터 생성 완료 {Xtr.shape} / {Xva.shape} ({time.time() - t0:.0f}s)")

    rng = np.random.default_rng(args.seed)
    model = NumpyCNN(seed=args.seed)
    xtr = ((Xtr - MEAN) / STD)[:, None].astype(np.float32)
    best, best_p = 0.0, None
    for ep in range(1, args.epochs + 1):
        lr = 1e-3 * (0.5 ** ((ep - 1) // 2))
        order = rng.permutation(len(xtr))
        losses, t1 = [], time.time()
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            xb = xtr[idx]
            # 소량의 위치 흔들림 증강 (±2px)
            sy, sx = rng.integers(-2, 3, 2)
            xb = np.roll(xb, (int(sy), int(sx)), axis=(2, 3))
            logits, cache = model.forward(xb, train=True, rng=rng)
            loss, d = model.loss_and_grad(logits, ytr[idx])
            model.adam_step(model.backward(d, cache), lr=lr)
            losses.append(loss)
        acc = float((model.predict(Xva) == yva).mean())
        print(f"[{ep}/{args.epochs}] loss {np.mean(losses):.4f} val(synthetic) {acc:.4f} lr {lr:.1e} "
              f"({time.time() - t1:.0f}s)")
        if acc > best:
            best, best_p = acc, {k: v.copy() for k, v in model.p.items()}
    model.p = best_p
    real = sklearn_digits_eval(model)
    print(f"합성 검증 정확도 {best:.4f}, sklearn 실제 손글씨(8x8) 참고 정확도 {real}")

    data = build_model_dict(model.to_layers(), meta={
        "source": "numpy-synthetic-bootstrap",
        "description": "임시 모델: 합성 손글씨(획 템플릿+글꼴)로 NumPy 학습. MNIST 학습 가중치로 교체 필요",
        "synthetic_val_accuracy": round(best, 4),
        "sklearn_digits_accuracy": None if real is None else round(real, 4),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    })
    out = write_weights_js(data, args.out)
    print(f"저장: {out} ({out.stat().st_size / 1024:.0f} KB), 총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
