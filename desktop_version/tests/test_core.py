"""PyTorch 없이 실행되는 테스트 (전처리, 가중치 포맷, NumPy 참조 구현).

    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from preprocess import area_resize, center_digit, round_half_up, to_gray  # noqa: E402
from weights_format import build_model_dict, numpy_forward, read_weights_js, write_weights_js  # noqa: E402


class PreprocessTest(unittest.TestCase):
    def test_round_half_up_matches_js(self):
        self.assertEqual([round_half_up(v) for v in (0.5, 1.5, 2.5, -0.5, -1.5)], [1, 2, 3, 0, -1])

    def test_area_resize_block_average(self):
        img = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        np.testing.assert_allclose(area_resize(img, 2, 2), [[1, 0], [0, 0.5]])

    def test_area_resize_preserves_mean(self):
        img = np.random.default_rng(0).random((37, 23))
        out = area_resize(img, 20, 12)
        self.assertAlmostEqual(out.mean(), img.mean(), places=10)

    def test_empty_returns_none(self):
        self.assertIsNone(center_digit(np.zeros((10, 10))))

    def test_centering_and_box(self):
        g = np.zeros((50, 50)); g[0:10, 0:5] = 1.0
        out = center_digit(g)
        self.assertEqual(out.shape, (28, 28))
        rows = np.nonzero(out.sum(1))[0]
        self.assertEqual(len(rows), 20)
        cy = (out.sum(1) * np.arange(28)).sum() / out.sum()
        cx = (out.sum(0) * np.arange(28)).sum() / out.sum()
        self.assertLessEqual(abs(cy - 13.5), 0.5)
        self.assertLessEqual(abs(cx - 13.5), 0.5)

    def test_to_gray_inverts_white_background(self):
        img = np.ones((5, 5)); img[2, 2] = 0
        g = to_gray(img)
        self.assertEqual(g[2, 2], 1.0)
        self.assertEqual(g[0, 0], 0.0)


class WeightsFormatTest(unittest.TestCase):
    def _tiny_layers(self, rng):
        return [
            {"type": "conv2d", "weight": rng.normal(size=(8, 1, 5, 5)), "bias": rng.normal(size=8)},
            {"type": "relu"}, {"type": "maxpool2d", "k": 2},
            {"type": "conv2d", "weight": rng.normal(size=(16, 8, 5, 5)) * 0.1, "bias": rng.normal(size=16)},
            {"type": "relu"}, {"type": "maxpool2d", "k": 2}, {"type": "flatten"},
            {"type": "linear", "weight": rng.normal(size=(64, 256)) * 0.1, "bias": rng.normal(size=64)},
            {"type": "relu"},
            {"type": "linear", "weight": rng.normal(size=(10, 64)) * 0.1, "bias": rng.normal(size=10)},
        ]

    def test_roundtrip(self):
        model = build_model_dict(self._tiny_layers(np.random.default_rng(0)), meta={"source": "test"})
        with tempfile.TemporaryDirectory() as d:
            p = write_weights_js(model, Path(d) / "w.js")
            self.assertTrue(p.read_text(encoding="utf-8").startswith("/*"))
            back = read_weights_js(p)
        self.assertEqual(back["layers"][0]["outC"], 8)
        self.assertEqual(len(back["layers"][7]["weight"]), 64 * 256)

    def test_numpy_forward_matches_numpy_cnn(self):
        """직렬화 참조 순전파 == 학습용 NumpyCNN 순전파 (레이아웃 일치 확인)."""
        from numpy_cnn import MEAN, STD, NumpyCNN
        net = NumpyCNN(seed=3)
        model = build_model_dict(net.to_layers(), meta={}, digits=10)
        x = np.random.default_rng(1).random((28, 28)).astype(np.float32)
        expected = net.forward(((x - MEAN) / STD)[None, None])[0][0]
        np.testing.assert_allclose(numpy_forward(model, x), expected, rtol=1e-4, atol=1e-4)

    def test_rejects_unknown_layer(self):
        with self.assertRaises(ValueError):
            build_model_dict([{"type": "softmax"}], meta={})


class ShippedWeightsTest(unittest.TestCase):
    def test_web_weights_are_valid(self):
        path = ROOT.parent / "web_version" / "model" / "weights.js"
        if not path.exists():
            self.skipTest("weights.js 없음")
        model = read_weights_js(path)
        logits = numpy_forward(model, np.zeros((28, 28)))
        self.assertEqual(logits.shape, (10,))


if __name__ == "__main__":
    unittest.main()
