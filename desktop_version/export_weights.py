"""학습된 PyTorch 체크포인트 → web_version/model/weights.js 로 내보내기.

사용 예:
    python export_weights.py                       # 기본 경로
    python export_weights.py --ckpt checkpoints/mnist_cnn.pt --out ../web_version/model/weights.js
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from torch import nn

from model import MnistCNN, load_checkpoint
from weights_format import DEFAULT_WEB_PATH, build_model_dict, write_weights_js

HERE = Path(__file__).resolve().parent


def model_to_layers(model: MnistCNN) -> list[dict]:
    """nn.Module 을 순회하며 웹 포맷 레이어 목록으로 변환 (Dropout 은 생략)."""
    layers = []
    for m in list(model.features) + list(model.classifier):
        if isinstance(m, nn.Conv2d):
            assert m.stride == (1, 1) and m.padding == (0, 0), "웹 엔진은 stride 1, padding 0 만 지원"
            layers.append({"type": "conv2d", "weight": m.weight.detach().cpu().numpy(),
                           "bias": m.bias.detach().cpu().numpy()})
        elif isinstance(m, nn.Linear):
            layers.append({"type": "linear", "weight": m.weight.detach().cpu().numpy(),
                           "bias": m.bias.detach().cpu().numpy()})
        elif isinstance(m, nn.ReLU):
            layers.append({"type": "relu"})
        elif isinstance(m, nn.MaxPool2d):
            layers.append({"type": "maxpool2d", "k": int(m.kernel_size)})
        elif isinstance(m, nn.Flatten):
            layers.append({"type": "flatten"})
        elif isinstance(m, nn.Dropout):
            continue
        else:
            raise TypeError(f"웹으로 내보낼 수 없는 레이어: {m}")
    return layers


def main(argv=None):
    p = argparse.ArgumentParser(description="PyTorch 가중치 → 웹 weights.js")
    p.add_argument("--ckpt", type=Path, default=HERE / "checkpoints" / "mnist_cnn.pt")
    p.add_argument("--out", type=Path, default=DEFAULT_WEB_PATH)
    args = p.parse_args(argv)

    model, meta = load_checkpoint(args.ckpt)
    data = build_model_dict(model_to_layers(model), meta={
        "source": meta.get("source", "pytorch-mnist"),
        "description": "PyTorch 로 MNIST 에서 학습한 모델",
        "test_accuracy": round(float(meta.get("test_acc", 0.0)), 4),
        "epoch": meta.get("epoch"),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    })
    out = write_weights_js(data, args.out)
    print(f"내보내기 완료: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
