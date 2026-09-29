# CLAUDE.md — web_version (순수 JavaScript, GitHub Pages)

브라우저에서 손글씨를 그리면 CNN으로 숫자를 추론하는 **정적 웹 앱**.
외부 라이브러리·CDN·번들러·npm 의존성 **없음**. 루트 `CLAUDE.md`의 계약을 먼저 읽을 것.

## 🔴 채점 필수 조건 (절대 변경 금지)
`index.html`의 `<body>` 바로 다음 첫 요소:
```html
<header id="student-info" class="student-info">학번: 2601949 | 이름: 송은비</header>
```
- 텍스트로 노출(이미지·CSS content·JS 생성 금지), 화면 맨 위(sticky) 유지.
- `node tests/run_tests.js`가 이 조건을 검사한다.

## 파일
| 파일 | 역할 |
|---|---|
| `index.html` | 화면 구조. 스크립트 로드 순서: weights → preprocess → nn → app |
| `css/style.css` | 스타일 (라이트/다크, 모바일 대응) |
| `js/preprocess.js` | MNIST 전처리 (`desktop_version/preprocess.py`와 동일 알고리즘) |
| `js/nn.js` | CNN 순전파 엔진: conv2d / relu / maxpool2d / flatten / linear |
| `js/app.js` | 캔버스 그리기(Pointer Events), 이미지 불러오기, 결과 표시 |
| `model/weights.js` | **자동 생성** 가중치 (`window.MNIST_MODEL`). 직접 수정 금지 |
| `tests/run_tests.js` | Node 테스트 (단위 + Python 참조 구현과 일치 + index.html 검사) |
| `tests/fixture.json` | `desktop_version/tools/make_js_fixture.py`가 생성 |
| `.nojekyll` | GitHub Pages의 Jekyll 처리 끄기 |

## 명령
```powershell
node tests/run_tests.js                 # 테스트 (Node 18+)
python -m http.server 8000              # 로컬 확인 → http://localhost:8000
```
`index.html`을 더블클릭(file://)해도 동작한다 (ES 모듈·fetch 미사용).

## 규칙
- 모든 JS는 UMD 형태 `(function(root, factory){...})` — 브라우저 전역 + Node `require` 둘 다 지원.
- ES 모듈(`type="module"`)과 `fetch()`로 가중치를 읽는 방식은 쓰지 않는다 (file:// 에서 깨짐).
- 경로는 모두 상대 경로 (GitHub Pages 하위 경로 `/<repo>/`에서 동작해야 함).
- 전처리를 바꾸면 Python 쪽도 같이 바꾸고 fixture 재생성 → 테스트.
- `model/weights.js`의 `meta.source`에 `bootstrap`이 들어 있으면 UI에 "임시 모델" 배지가 뜬다.
  실제 MNIST 가중치로 교체하려면 `desktop_version`에서 `train.py` → `export_weights.py`.

## 배포 (GitHub Pages)
- 저장소 루트의 `.github/workflows/pages.yml`이 `web_version/` 폴더만 Pages로 배포한다.
- 저장소 Settings → Pages → Source: **GitHub Actions** 선택.
- 또는 이 폴더 내용만 별도 저장소/`gh-pages` 브랜치 루트에 올려도 된다.
