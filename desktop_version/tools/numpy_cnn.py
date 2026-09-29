"""model.MnistCNN 과 동일 구조의 NumPy 구현 (순전파 + 역전파 + Adam).

PyTorch 를 설치할 수 없는 환경에서 임시(부트스트랩) 가중치를 만들기 위한 용도.
가중치 배치는 PyTorch 와 동일: conv (O,C,k,k), linear (out,in).
"""
from __future__ import annotations

import numpy as np

MEAN, STD = 0.1307, 0.3081


def im2col(x: np.ndarray, k: int) -> np.ndarray:
    """(N,C,H,W) → (N, OH*OW, C*k*k)  (C,kh,kw 순서 = PyTorch 가중치 평탄화 순서)."""
    N, C, H, W = x.shape
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))  # N,C,OH,OW,k,k
    return np.ascontiguousarray(win.transpose(0, 2, 3, 1, 4, 5)).reshape(N, (H - k + 1) * (W - k + 1), C * k * k)


class NumpyCNN:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)

        def he(shape, fan_in):
            return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)

        self.p = {
            "c1w": he((8, 1, 5, 5), 25), "c1b": np.zeros(8, np.float32),
            "c2w": he((16, 8, 5, 5), 200), "c2b": np.zeros(16, np.float32),
            "f1w": he((64, 256), 256), "f1b": np.zeros(64, np.float32),
            "f2w": he((10, 64), 64) * 0.5, "f2b": np.zeros(10, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    # ---------- 순전파 ----------
    @staticmethod
    def _conv(x, w, b):
        N, _, H, W = x.shape
        O, _, k, _ = w.shape
        cols = im2col(x, k)
        out = cols @ w.reshape(O, -1).T + b  # N,P,O
        return out.transpose(0, 2, 1).reshape(N, O, H - k + 1, W - k + 1), cols

    @staticmethod
    def _pool(x):
        N, C, H, W = x.shape
        r = x.reshape(N, C, H // 2, 2, W // 2, 2)
        out = r.max(axis=(3, 5))
        mask = r == out[:, :, :, None, :, None]
        return out, mask

    def forward(self, x, train: bool = False, drop: float = 0.25, rng=None):
        """x: (N,1,28,28) 정규화된 입력 → logits (N,10)."""
        p, c = self.p, {}
        z1, c["cols1"] = self._conv(x, p["c1w"], p["c1b"])
        a1 = np.maximum(z1, 0)
        c["z1"] = z1
        h1, c["mask1"] = self._pool(a1)
        z2, c["cols2"] = self._conv(h1, p["c2w"], p["c2b"])
        a2 = np.maximum(z2, 0)
        c["z2"], c["h1_shape"] = z2, h1.shape
        h2, c["mask2"] = self._pool(a2)
        f = h2.reshape(len(x), -1)
        z3 = f @ p["f1w"].T + p["f1b"]
        a3 = np.maximum(z3, 0)
        if train and drop > 0:
            keep = (rng.random(a3.shape) >= drop).astype(np.float32) / (1 - drop)
            a3 = a3 * keep
            c["keep"] = keep
        logits = a3 @ p["f2w"].T + p["f2b"]
        c.update(f=f, z3=z3, a3=a3, h2_shape=h2.shape)
        return logits, c

    # ---------- 역전파 ----------
    def backward(self, dlogits, c):
        p, g = self.p, {}
        N = len(dlogits)
        g["f2w"] = dlogits.T @ c["a3"]
        g["f2b"] = dlogits.sum(0)
        da3 = dlogits @ p["f2w"]
        if "keep" in c:
            da3 = da3 * c["keep"]
        dz3 = da3 * (c["z3"] > 0)
        g["f1w"] = dz3.T @ c["f"]
        g["f1b"] = dz3.sum(0)
        dh2 = (dz3 @ p["f1w"]).reshape(c["h2_shape"])

        da2 = self._unpool(dh2, c["mask2"])
        dz2 = da2 * (c["z2"] > 0)
        d2 = dz2.reshape(N, 16, -1).transpose(0, 2, 1)  # N,P,O
        g["c2w"] = np.einsum("npo,npk->ok", d2, c["cols2"]).reshape(p["c2w"].shape)
        g["c2b"] = d2.sum((0, 1))
        dcols = d2 @ p["c2w"].reshape(16, -1)  # N,P,C*k*k
        dh1 = self._col2im(dcols, c["h1_shape"], 5)

        da1 = self._unpool(dh1, c["mask1"])
        dz1 = da1 * (c["z1"] > 0)
        d1 = dz1.reshape(N, 8, -1).transpose(0, 2, 1)
        g["c1w"] = np.einsum("npo,npk->ok", d1, c["cols1"]).reshape(p["c1w"].shape)
        g["c1b"] = d1.sum((0, 1))
        return g

    @staticmethod
    def _unpool(dout, mask):
        N, C, H, W = dout.shape
        return (mask * dout[:, :, :, None, :, None]).reshape(N, C, H * 2, W * 2)

    @staticmethod
    def _col2im(dcols, shape, k):
        N, C, H, W = shape
        OH, OW = H - k + 1, W - k + 1
        d = dcols.reshape(N, OH, OW, C, k, k)
        dx = np.zeros(shape, dtype=dcols.dtype)
        for i in range(k):
            for j in range(k):
                dx[:, :, i:i + OH, j:j + OW] += d[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dx

    # ---------- 학습 ----------
    @staticmethod
    def loss_and_grad(logits, y):
        z = logits - logits.max(1, keepdims=True)
        e = np.exp(z)
        prob = e / e.sum(1, keepdims=True)
        N = len(y)
        loss = -np.log(prob[np.arange(N), y] + 1e-12).mean()
        d = prob.copy()
        d[np.arange(N), y] -= 1
        return loss, d / N

    def adam_step(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def predict(self, x28_batch: np.ndarray) -> np.ndarray:
        x = ((x28_batch - MEAN) / STD)[:, None].astype(np.float32)
        out = []
        for i in range(0, len(x), 512):
            out.append(self.forward(x[i:i + 512])[0])
        return np.concatenate(out).argmax(1)

    def to_layers(self) -> list[dict]:
        p = self.p
        return [
            {"type": "conv2d", "weight": p["c1w"], "bias": p["c1b"]}, {"type": "relu"}, {"type": "maxpool2d", "k": 2},
            {"type": "conv2d", "weight": p["c2w"], "bias": p["c2b"]}, {"type": "relu"}, {"type": "maxpool2d", "k": 2},
            {"type": "flatten"},
            {"type": "linear", "weight": p["f1w"], "bias": p["f1b"]}, {"type": "relu"},
            {"type": "linear", "weight": p["f2w"], "bias": p["f2b"]},
        ]
