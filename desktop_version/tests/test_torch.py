"""PyTorch 가 설치된 환경에서만 실행되는 테스트 (없으면 자동 skip).

    python -m unittest discover -s tests -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "PyTorch 미설치")
class TorchModelTest(unittest.TestCase):
    def test_output_shape(self):
        from model import MnistCNN
        out = MnistCNN()(torch.zeros(4, 1, 28, 28))
        self.assertEqual(tuple(out.shape), (4, 10))

    def test_one_step_reduces_loss(self):
        from model import MnistCNN
        torch.manual_seed(0)
        m = MnistCNN(dropout=0.0)
        x, y = torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,))
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        losses = []
        for _ in range(5):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(m(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])

    def test_export_matches_torch_forward(self):
        """export → weights.js → NumPy 참조 순전파 결과가 PyTorch 와 일치 (JS 는 fixture 로 별도 검증)."""
        from export_weights import model_to_layers
        from model import MEAN, STD, MnistCNN
        from weights_format import build_model_dict, numpy_forward, read_weights_js, write_weights_js

        torch.manual_seed(1)
        m = MnistCNN().eval()
        with tempfile.TemporaryDirectory() as d:
            p = write_weights_js(build_model_dict(model_to_layers(m), meta={"source": "t"}, digits=8), Path(d) / "w.js")
            data = read_weights_js(p)
        x = np.random.default_rng(0).random((28, 28)).astype(np.float32)
        with torch.no_grad():
            ref = m(((torch.from_numpy(x) - MEAN) / STD).view(1, 1, 28, 28))[0].numpy()
        np.testing.assert_allclose(numpy_forward(data, x), ref, rtol=1e-4, atol=1e-4)

    def test_checkpoint_roundtrip(self):
        from model import MnistCNN, load_checkpoint
        m = MnistCNN()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.pt"
            torch.save({"state_dict": m.state_dict(), "test_acc": 0.5}, p)
            m2, meta = load_checkpoint(p)
        self.assertEqual(meta["test_acc"], 0.5)
        for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
