# 📄 HangulPDF AI Converter

한글로 작성된 PDF 문서를 AI가 쉽게 활용할 수 있도록 자동 변환하고, 요약하고, 질문 응답 및 공유 가능한 형태로 저장하는 웹 애플리케이션입니다.

## 🚀 주요 기능

- **PDF 텍스트 추출**: 한글 PDF에서 텍스트를 정확하게 추출
- **AI 요약**: OpenAI GPT를 활용한 문서 요약 생성
- **질문-답변 생성**: 문서 내용 기반 Q&A 자동 생성
- **텍스트 정제**: 추출된 텍스트의 품질 향상
- **다양한 내보내기**: 텍스트, 요약 파일 다운로드
- **AI 모델 연동**: ChatGPT, Gemini, Grok 등과 쉬운 연동

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **Backend**: FastAPI (선택적)
- **PDF 처리**: pdfplumber
- **AI 모델**: OpenAI GPT-3.5/GPT-4
- **언어 처리**: kss, hanspell
- **클라우드 연동**: Google Drive API

## 📦 설치 및 실행

### 1. 로컬 실행

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/hangulpdf-ai.git
cd hangulpdf-ai

# 의존성 설치
pip install -r requirements.txt

# Streamlit 앱 실행
streamlit run streamlit_app.py
```

### 2. Streamlit Cloud 배포

1. 이 저장소를 GitHub에 fork 또는 clone
2. [Streamlit Cloud](https://streamlit.io/cloud)에 로그인
3. "New app" 클릭 후 이 저장소 선택
4. App path를 `streamlit_app.py`로 설정
5. Advanced settings에서 Secrets 설정:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "your-openai-api-key-here"
GDRIVE_SERVICE_JSON = "your-google-drive-service-account-json-here"
```

6. Deploy 클릭!

## 🔑 환경 변수 설정

### 필수 설정
- `OPENAI_API_KEY`: OpenAI API 키 (GPT 모델 사용)

### 선택적 설정
- `GDRIVE_SERVICE_JSON`: Google Drive 업로드용 서비스 계정 JSON

## 📖 사용법

1. **PDF 업로드**: 한글 PDF 파일을 업로드
2. **변환 옵션 선택**: 텍스트 추출, 요약 생성, Q&A 생성 등
3. **변환 실행**: AI가 문서를 분석하고 변환
4. **결과 확인**: 추출된 텍스트, 요약, 질문-답변 확인
5. **내보내기**: 결과를 파일로 다운로드하거나 다른 AI 모델에 연동

## 🤖 AI 모델 연동 방법

### ChatGPT
추출된 텍스트를 복사하여 ChatGPT에 직접 붙여넣기

### Gemini
Google AI Studio에서 추출된 텍스트 활용

### Grok (X AI)
X 플랫폼의 Grok에 텍스트 입력

## 🔧 개발자 정보

### 프로젝트 구조
```
hangulpdf-ai/
├── streamlit_app.py          # Streamlit 메인 앱
├── main.py                   # FastAPI 서버 (선택적)
├── requirements.txt          # Python 의존성
├── modules/                  # 핵심 모듈들
│   ├── converter.py         # PDF 변환 로직
│   ├── gpt_summary.py       # GPT 요약 생성
│   ├── gpt_qa.py           # GPT Q&A 생성
│   ├── text_cleaner.py     # 텍스트 정제
│   ├── drive_uploader.py   # Google Drive 업로드
│   └── export_grok.py      # Grok 연동
└── .streamlit/
    └── secrets.toml         # 환경 변수 (로컬용)
```

## 🚀 배포 옵션

### Streamlit Cloud (추천)
- 무료 호스팅
- GitHub 연동 자동 배포
- 간편한 환경 변수 관리

### Hugging Face Spaces
- 무료 호스팅
- GPU 지원 (유료)
- 커뮤니티 공유 용이

### Heroku/Railway
- 더 많은 커스터마이징 가능
- 데이터베이스 연동 용이

## 📝 라이선스

MIT License

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 문의

프로젝트 관련 문의사항이 있으시면 Issues를 통해 연락해주세요.

---

**HangulPDF AI Converter** - 한글 PDF 문서를 AI 시대에 맞게 변환하는 도구 🚀

