"""MNIST CNN 학습 스크립트.

사용 예 (desktop_version 폴더에서):
    python train.py                 # 기본 8 epoch
    python train.py --epochs 3 --batch-size 128
결과: checkpoints/mnist_cnn.pt (최고 테스트 정확도 모델), checkpoints/metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MEAN, STD, MnistCNN, pick_device

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MNIST CNN 학습")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--out", type=Path, default=HERE / "checkpoints" / "mnist_cnn.pt")
    p.add_argument("--no-augment", action="store_true", help="학습 데이터 증강 끄기")
    p.add_argument("--workers", type=int, default=0, help="DataLoader 워커 수 (Windows 는 0 권장)")
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loaders(args):
    normalize = transforms.Normalize((MEAN,), (STD,))
    train_tf = [transforms.ToTensor(), normalize]
    if not args.no_augment:
        # 캔버스 손글씨는 MNIST 보다 기울기/위치 편차가 크므로 약한 어파인 증강
        train_tf.insert(0, transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8))
    train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transforms.Compose(train_tf))
    test_set = datasets.MNIST(args.data_dir, train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=args.workers)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        seen += y.size(0)
    return total_loss / seen, correct / seen


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device()
    print(f"장치: {device}")

    train_loader, test_loader = make_loaders(args)
    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    history, best_acc = [], 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "test_loss": te_loss, "test_acc": te_acc})
        mark = ""
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({"state_dict": model.state_dict(), "test_acc": te_acc, "epoch": epoch,
                        "source": "pytorch-mnist"}, args.out)
            mark = "  ← 저장"
        print(f"[{epoch:02d}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f} ({time.time() - t0:.1f}s){mark}")

    (args.out.parent / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"완료. 최고 테스트 정확도 {best_acc:.4f} → {args.out}")
    print("웹 버전에 반영하려면: python export_weights.py")


if __name__ == "__main__":
    main()
