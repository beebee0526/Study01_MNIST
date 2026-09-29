# CLAUDE.md — MNIST 손글씨 인식 프로젝트 (루트)

학번: 2601949 | 이름: 송은비

MNIST 손글씨 숫자 인식을 **데스크톱(PyTorch)** 과 **웹(순수 JS)** 두 버전으로 구현한다.
데스크톱에서 학습한 가중치를 웹으로 내보내 GitHub Pages에서 정적으로 추론한다.

## 구조
```
.
├── CLAUDE.md              # (이 파일) 프로젝트 공통 규칙
├── CLAUDE_전역.md          # 개인 전역 지침 (~/.claude/CLAUDE.md 로 복사해서 사용)
├── README.md
├── .github/workflows/pages.yml   # web_version/ 을 GitHub Pages 로 배포
├── desktop_version/       # PyTorch 학습·추론·가중치 내보내기  → desktop_version/CLAUDE.md
└── web_version/           # 순수 JS 추론 웹 앱 (정적)           → web_version/CLAUDE.md
```

## 두 버전 사이의 계약 (가장 중요)
아래 세 가지는 양쪽이 **반드시 동일**해야 한다. 하나를 바꾸면 양쪽을 함께 바꾸고 일치 테스트를 다시 돌린다.

| 항목 | desktop_version | web_version |
|---|---|---|
| 모델 구조 | `model.py` (`MnistCNN`) | `js/nn.js` (레이어 연산) |
| 가중치 포맷 `mnist-cnn-v1` | `weights_format.py` (작성) | `model/weights.js` (읽기) |
| 전처리 (bbox → 20px 면적평균 → 무게중심 정렬) | `preprocess.py` | `js/preprocess.js` |

- 모델: Conv(1→8,5)→ReLU→Pool2 → Conv(8→16,5)→ReLU→Pool2 → FC(256→64)→ReLU→FC(64→10)
- 정규화: mean 0.1307, std 0.3081 / 입력 28×28, 흰 글씨·검은 배경
- 붓 굵기: 280×280 캔버스 기준 18px (`draw_app.py`, `app.js`)

## 표준 작업 흐름
```powershell
cd desktop_version
python train.py                      # MNIST 학습 → checkpoints/mnist_cnn.pt
python export_weights.py             # → ../web_version/model/weights.js
python tools/make_js_fixture.py      # JS 일치 테스트용 fixture 갱신
python -m unittest discover -s tests -v
cd ../web_version
node tests/run_tests.js              # 전처리·logits 가 Python 과 일치하는지 확인
```

## 규칙
- **채점 필수**: `web_version/index.html`의 `<body>` 첫 요소는 반드시
  `<header id="student-info">학번: 2601949 | 이름: 송은비</header>` 이다. 삭제·이동·문구 변경 금지.
  (`web_version/tests/run_tests.js`가 이를 검사한다)
- `web_version/`에는 외부 라이브러리·CDN·빌드 도구를 쓰지 않는다.
- `web_version/model/weights.js`는 자동 생성 파일이다. 손으로 고치지 않는다.
- 학습 데이터(`desktop_version/data/`)와 체크포인트(`*.pt`)는 커밋하지 않는다(.gitignore).
- 완료 보고 전: Python 단위 테스트 + Node 테스트 모두 통과 확인.
