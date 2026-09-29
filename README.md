# MNIST 손글씨 인식 — Desktop(PyTorch) & Web(순수 JS)

학번: 2601949 | 이름: 송은비

| 버전 | 기술 | 내용 |
|---|---|---|
| `desktop_version/` | Python, PyTorch | CNN 학습, CLI/GUI 추론, 웹용 가중치 내보내기 |
| `web_version/` | HTML/CSS/순수 JavaScript | 브라우저 캔버스에 그린 숫자를 CNN으로 추론 (GitHub Pages 정적 배포) |

## 빠른 시작 (Windows PowerShell)

### 1) 데스크톱: 학습 → 추론
```powershell
cd desktop_version
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py                  # MNIST 자동 다운로드 후 학습 (CPU 약 수 분)
python predict.py --mnist-test 20
python draw_app.py               # 마우스로 그려서 인식하는 GUI
```

### 2) 웹으로 가중치 내보내기
```powershell
python export_weights.py         # → web_version/model/weights.js
python tools/make_js_fixture.py  # JS 테스트 기준값 갱신
cd ..\web_version
node tests/run_tests.js
```

### 3) 웹 실행 / 배포
- 로컬: `web_version/index.html` 더블클릭, 또는 `python -m http.server 8000` 후 http://localhost:8000
- GitHub Pages: 저장소에 push → Settings → Pages → Source를 **GitHub Actions**로 설정
  (`.github/workflows/pages.yml`이 `web_version/`을 배포)

## 모델
```
입력 1×28×28 → Conv5×5(8) → ReLU → MaxPool2 → Conv5×5(16) → ReLU → MaxPool2
            → FC 256→64 → ReLU → (Dropout) → FC 64→10
```
파라미터 약 2만 개, 가중치 파일 약 190KB. MNIST에서 보통 98.5~99% 테스트 정확도.

## 참고: 동봉된 "임시 모델"
처음 포함된 `web_version/model/weights.js`는 네트워크가 막힌 개발 환경에서 만든 **임시 가중치**다
(`desktop_version/tools/bootstrap_synthetic.py`: NumPy로 같은 CNN을 합성 손글씨 데이터로 학습).
웹 화면에 "임시 모델" 배지가 보이면 위 1)~2) 단계를 실행해 실제 MNIST 가중치로 교체하면 된다.
