# CLAUDE.md — desktop_version (PyTorch)

MNIST CNN 학습, 추론(CLI·GUI), 웹용 가중치 내보내기를 담당한다.
루트 `CLAUDE.md`의 "두 버전 사이의 계약"을 먼저 읽을 것.

## 파일
| 파일 | 역할 | torch 필요 |
|---|---|---|
| `model.py` | `MnistCNN` 정의, `load_checkpoint`, 장치 선택 | O |
| `train.py` | MNIST 다운로드·학습·평가, 최고 모델 저장 | O |
| `predict.py` | 이미지 파일 / MNIST 테스트셋 추론 (CLI) | O |
| `draw_app.py` | tkinter 손글씨 그리기 GUI | O |
| `export_weights.py` | 체크포인트 → `../web_version/model/weights.js` | O |
| `preprocess.py` | MNIST 스타일 전처리 (JS와 동일 알고리즘) | X |
| `weights_format.py` | 웹 가중치 포맷 작성/읽기 + NumPy 참조 순전파 | X |
| `tools/numpy_cnn.py` | 같은 구조의 NumPy 구현(역전파 포함) | X |
| `tools/synth_data.py` | 합성 손글씨(획 템플릿+글꼴) 생성기 | X |
| `tools/bootstrap_synthetic.py` | torch/MNIST 없이 임시 가중치 생성 | X |
| `tools/make_js_fixture.py` | JS 일치 테스트용 `web_version/tests/fixture.json` 생성 | X |
| `tests/test_core.py` | 전처리·포맷 테스트 | X |
| `tests/test_torch.py` | 모델·export 일치 테스트 (torch 없으면 skip) | O |

## 명령 (Windows PowerShell, 이 폴더에서)
```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
python train.py --epochs 8           # checkpoints/mnist_cnn.pt, metrics.json
python predict.py --mnist-test 20
python predict.py my_digit.png
python draw_app.py
python export_weights.py
python tools/make_js_fixture.py
python -m unittest discover -s tests -v
```

## 규칙
- 모델 구조를 바꾸면: `export_weights.model_to_layers`, `tools/numpy_cnn.py`,
  `web_version/js/nn.js`의 지원 레이어, `web_version/tests/run_tests.js`의 구조 검사를 함께 확인한다.
- `preprocess.py`를 바꾸면 `web_version/js/preprocess.js`도 똑같이 바꾸고 fixture를 다시 만든다.
  반올림은 `round_half_up`(JS `Math.round`와 동일)을 쓴다. Python `round()` 금지(banker's rounding).
- 가중치를 내보낸 뒤에는 항상 `tools/make_js_fixture.py` → `node ../web_version/tests/run_tests.js`.
- Windows에서는 DataLoader `--workers 0`(기본값)을 유지한다.
- `data/`, `checkpoints/`는 git에 올리지 않는다.
- `tools/bootstrap_synthetic.py`는 네트워크가 막힌 환경용 **임시** 경로다. 최종 제출물은 `train.py` 결과를 쓴다.
