"""MNIST CNN 모델 정의.

구조 (web_version/js/nn.js 가 동일 연산을 순수 JS 로 재현):
    입력 1x28x28
    Conv2d(1→8, 5x5)  → ReLU → MaxPool(2)   : 8x12x12
    Conv2d(8→16, 5x5) → ReLU → MaxPool(2)   : 16x4x4
    Flatten(256) → Linear(256→64) → ReLU → Dropout → Linear(64→10)

구조를 바꾸면 export_weights.py 의 레이어 매핑과 웹 쪽 지원 레이어를 확인할 것.
(Dropout 은 추론 시 항등이므로 내보내지 않는다)
"""
import torch
from torch import nn

MEAN, STD = 0.1307, 0.3081
NUM_CLASSES = 10


class MnistCNN(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path, device=None) -> tuple[MnistCNN, dict]:
    """train.py 가 저장한 체크포인트를 읽어 (eval 모드 모델, 메타정보) 반환."""
    device = device or torch.device("cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MnistCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}
