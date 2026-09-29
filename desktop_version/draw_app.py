"""데스크톱 손글씨 인식 GUI (tkinter, 표준 라이브러리).

    python draw_app.py [--ckpt checkpoints/mnist_cnn.pt]

마우스로 숫자를 그리면 손을 뗄 때마다 자동으로 인식한다.
"""
from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from model import load_checkpoint
from predict import predict_array
from preprocess import center_digit

HERE = Path(__file__).resolve().parent
CANVAS = 280
BRUSH = 18  # web_version 과 동일한 굵기


class DrawApp:
    def __init__(self, root: tk.Tk, model, meta: dict):
        self.model = model
        root.title("MNIST 손글씨 인식 (Desktop)")
        root.resizable(False, False)

        tk.Label(root, text="학번: 2601949 | 이름: 송은비", font=("Malgun Gothic", 12, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(8, 2))
        tk.Label(root, text=f"모델 테스트 정확도: {meta.get('test_acc', 0) * 100:.2f}%", fg="#555").grid(
            row=1, column=0, columnspan=2)

        self.canvas = tk.Canvas(root, width=CANVAS, height=CANVAS, bg="black", cursor="pencil")
        self.canvas.grid(row=2, column=0, padx=10, pady=10)
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_move)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.predict())

        side = tk.Frame(root)
        side.grid(row=2, column=1, padx=10, sticky="n")
        self.result = tk.Label(side, text="?", font=("Arial", 64, "bold"), width=2)
        self.result.pack(pady=(10, 4))
        self.bars = []
        for d in range(10):
            row = tk.Frame(side)
            row.pack(anchor="w")
            tk.Label(row, text=str(d), width=2).pack(side="left")
            bar = tk.Canvas(row, width=150, height=14, bg="#eee", highlightthickness=0)
            bar.pack(side="left", pady=1)
            self.bars.append(bar)
        tk.Button(side, text="지우기", command=self.clear, width=12).pack(pady=10)

        self.image = Image.new("L", (CANVAS, CANVAS), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last = None

    def on_down(self, e):
        self.last = (e.x, e.y)
        self._dot(e.x, e.y)

    def on_move(self, e):
        if self.last is None:
            self.last = (e.x, e.y)
        x0, y0 = self.last
        self.canvas.create_line(x0, y0, e.x, e.y, fill="white", width=BRUSH, capstyle="round", smooth=True)
        self.draw.line([x0, y0, e.x, e.y], fill=255, width=BRUSH)
        self._dot(e.x, e.y)
        self.last = (e.x, e.y)

    def _dot(self, x, y):
        r = BRUSH / 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, CANVAS, CANVAS], fill=0)
        self.result.config(text="?")
        for bar in self.bars:
            bar.delete("all")

    def predict(self):
        self.last = None
        arr = center_digit(np.asarray(self.image, dtype=np.float32) / 255.0)
        if arr is None:
            return
        probs = predict_array(self.model, arr)
        best = int(probs.argmax())
        self.result.config(text=str(best))
        for d, bar in enumerate(self.bars):
            bar.delete("all")
            bar.create_rectangle(0, 0, int(150 * probs[d]), 14, fill="#2563eb" if d == best else "#94a3b8", width=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=HERE / "checkpoints" / "mnist_cnn.pt")
    args = p.parse_args()
    model, meta = load_checkpoint(args.ckpt)
    root = tk.Tk()
    DrawApp(root, model, meta)
    root.mainloop()


if __name__ == "__main__":
    main()
