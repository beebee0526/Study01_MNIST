"""학습된 모델로 추론.

사용 예:
    python predict.py my_digit.png other.jpg     # 이미지 파일 추론 (흰 바탕/검은 글씨도 자동 처리)
    python predict.py --mnist-test 20            # MNIST 테스트셋 앞 20장으로 확인
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model import MEAN, STD, load_checkpoint
from preprocess import image_to_mnist

HERE = Path(__file__).resolve().parent


def to_tensor(img28: np.ndarray) -> torch.Tensor:
    x = (torch.from_numpy(np.asarray(img28, dtype=np.float32)) - MEAN) / STD
    return x.view(1, 1, 28, 28)


@torch.no_grad()
def predict_array(model, img28: np.ndarray) -> np.ndarray:
    """28x28 [0,1] 배열 → 10개 클래스 확률."""
    return torch.softmax(model(to_tensor(img28)), dim=1)[0].cpu().numpy()


def predict_image(model, path: Path) -> np.ndarray | None:
    from PIL import Image
    arr = image_to_mnist(Image.open(path))
    return None if arr is None else predict_array(model, arr)


def format_top(probs: np.ndarray, k: int = 3) -> str:
    order = np.argsort(probs)[::-1][:k]
    return ", ".join(f"{d}({probs[d] * 100:.1f}%)" for d in order)


def main(argv=None):
    p = argparse.ArgumentParser(description="MNIST 추론")
    p.add_argument("images", nargs="*", type=Path)
    p.add_argument("--ckpt", type=Path, default=HERE / "checkpoints" / "mnist_cnn.pt")
    p.add_argument("--mnist-test", type=int, default=0, metavar="N", help="MNIST 테스트셋 N장 추론")
    args = p.parse_args(argv)

    model, meta = load_checkpoint(args.ckpt)
    print(f"모델: {args.ckpt} (테스트 정확도 {meta.get('test_acc', 0):.4f})")

    for path in args.images:
        probs = predict_image(model, path)
        if probs is None:
            print(f"{path}: 글씨를 찾지 못했습니다.")
        else:
            print(f"{path}: 예측 {int(probs.argmax())}  [{format_top(probs)}]")

    if args.mnist_test:
        from torchvision import datasets
        test = datasets.MNIST(HERE / "data", train=False, download=True)
        correct = 0
        for i in range(args.mnist_test):
            img, label = test[i]
            arr = np.asarray(img, dtype=np.float32) / 255.0  # MNIST 는 이미 전처리된 상태
            pred = int(predict_array(model, arr).argmax())
            correct += pred == label
            print(f"#{i:04d} 정답 {label} 예측 {pred} {'O' if pred == label else 'X'}")
        print(f"정확도 {correct}/{args.mnist_test}")

    if not args.images and not args.mnist_test:
        p.print_help()


if __name__ == "__main__":
    main()
