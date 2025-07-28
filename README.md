# 🔄 HangulPDF AI Converter

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hangulpdf-ai.streamlit.app)

한글 PDF 문서를 AI가 쉽게 활용할 수 있도록 자동 변환하고, 요약하고, 질문 응답 및 공유 가능한 형태로 저장하는 웹 애플리케이션입니다.

## 🚀 라이브 데모

**영구 배포 URL**: https://hangulpdf-ai.streamlit.app

## ✨ 주요 기능

### 📄 **PDF 처리**
- **텍스트 추출**: PyPDF2를 사용한 빠른 텍스트 추출
- **고급 OCR**: 이미지 기반 PDF, 표, 스캔 문서 지원
- **한글 특화**: 한글 문서 최적화된 처리

### 🤖 **AI 분석**
- **자동 AI 분석**: ChatGPT, Gemini, Grok 동시 분석
- **구조화된 요약**: 6단계 체계적 문서 분석
- **PDF 보고서 생성**: 분석 결과를 PDF로 자동 생성
- **ZIP 패키지**: 원본 + 분석 결과 일괄 다운로드

### 🎨 **사용자 경험**
- **모바일 반응형**: 모든 디바이스 지원
- **실시간 진행률**: 처리 상태 실시간 표시
- **한글 UI**: 완전한 한글 인터페이스

## 🛠️ 로컬 설치 및 실행

### 1. 저장소 클론
```bash
git clone https://github.com/yhun1542/hangulpdf-ai.git
cd hangulpdf-ai
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 환경 설정
`.streamlit/secrets.toml` 파일 생성:
```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

### 4. 앱 실행
```bash
streamlit run streamlit_app.py
```

## 🌐 Streamlit Cloud 배포

### 1. GitHub 저장소 연결
1. [Streamlit Cloud](https://streamlit.io/cloud) 접속
2. GitHub 계정으로 로그인
3. 이 저장소 선택

### 2. 환경 변수 설정
Streamlit Cloud 대시보드에서 다음 설정:
```
OPENAI_API_KEY = your-openai-api-key
```

### 3. 배포 완료
- **앱 진입점**: `streamlit_app.py`
- **Python 버전**: 3.11
- **자동 배포**: GitHub 푸시 시 자동 업데이트

## 📱 사용 방법

### **기본 사용**
1. 🔗 [앱 접속](https://hangulpdf-ai.streamlit.app)
2. 🔑 OpenAI API 키 입력 (좌측 사이드바)
3. 📄 PDF 파일 업로드
4. ⚙️ 변환 옵션 선택
5. 🚀 변환 시작

### **자동 AI 분석**
1. ✅ "자동 AI 분석 및 ZIP 다운로드" 체크
2. 📤 PDF 업로드 후 변환 시작
3. 🤖 ChatGPT, Gemini, Grok 자동 분석
4. 📦 완성된 ZIP 파일 다운로드

## 🔧 기술 스택

### **Backend**
- **Streamlit**: 웹 애플리케이션 프레임워크
- **PyPDF2**: PDF 텍스트 추출
- **pytesseract**: OCR 엔진
- **OpenCV**: 이미지 전처리

### **AI & PDF 생성**
- **OpenAI GPT**: 문서 분석 및 요약
- **WeasyPrint**: 고품질 PDF 생성
- **ReportLab**: 전문적 PDF 레포트
- **FPDF**: 기본 PDF 생성

### **배포**
- **Streamlit Cloud**: 영구 배포 플랫폼
- **GitHub**: 소스 코드 관리 및 CI/CD

## 📊 시스템 요구사항

- **Python**: 3.11+
- **메모리**: 최소 512MB
- **디스크**: 100MB 여유 공간
- **네트워크**: OpenAI API 접속 필요

## 🤝 기여하기

1. Fork 저장소
2. 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙋‍♂️ 지원

- **이슈 리포트**: [GitHub Issues](https://github.com/yhun1542/hangulpdf-ai/issues)
- **기능 요청**: [GitHub Discussions](https://github.com/yhun1542/hangulpdf-ai/discussions)

---

**Made with ❤️ for Korean PDF documents**

