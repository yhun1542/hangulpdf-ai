import streamlit as st
import requests
import json
import os
from io import BytesIO
import base64
import time

# OCR 및 이미지 처리를 위한 라이브러리
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# 진행률 표시를 위한 함수
def show_progress(progress_text, progress_value):
    """진행률을 표시하는 함수"""
    progress_bar = st.progress(progress_value)
    status_text = st.empty()
    status_text.text(progress_text)
    return progress_bar, status_text

# OCR 텍스트 추출 함수
def extract_text_with_ocr(pdf_bytes):
    """OCR을 사용하여 이미지 기반 PDF에서 텍스트 추출"""
    if not OCR_AVAILABLE:
        return "OCR 라이브러리가 설치되지 않았습니다."
    
    try:
        # PDF를 이미지로 변환
        images = convert_from_bytes(pdf_bytes, dpi=300)
        extracted_text = ""
        
        for i, image in enumerate(images):
            # 이미지 전처리 (OCR 정확도 향상)
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 노이즈 제거 및 대비 향상
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # OCR 실행 (한국어 + 영어)
            custom_config = r'--oem 3 --psm 6 -l kor+eng'
            text = pytesseract.image_to_string(denoised, config=custom_config)
            
            if text.strip():
                extracted_text += f"\n--- 페이지 {i+1} ---\n{text}\n"
        
        return extracted_text
    except Exception as e:
        return f"OCR 처리 중 오류: {str(e)}"

# 로컬 PDF 처리 함수 (OCR 및 진행률 표시 추가)
def process_pdf_locally(request_data, progress_callback=None):
    """PDF를 로컬에서 직접 처리하는 함수 (OCR 및 진행률 표시 포함)"""
    try:
        import pdfplumber
        import openai
        
        if progress_callback:
            progress_callback("PDF 파일 디코딩 중...", 0.1)
        
        # base64 디코딩
        file_content = base64.b64decode(request_data['file_content'])
        
        if progress_callback:
            progress_callback("텍스트 추출 중...", 0.2)
        
        # PDF 텍스트 추출 (기본 방식)
        extracted_text = ""
        try:
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
        except Exception as e:
            st.warning(f"기본 텍스트 추출 실패: {str(e)}")
        
        # 텍스트가 충분하지 않으면 OCR 시도
        if len(extracted_text.strip()) < 100 and OCR_AVAILABLE:
            if progress_callback:
                progress_callback("OCR을 사용한 텍스트 추출 중...", 0.4)
            
            ocr_text = extract_text_with_ocr(file_content)
            if ocr_text and len(ocr_text.strip()) > len(extracted_text.strip()):
                extracted_text = ocr_text
                st.info("📷 OCR을 사용하여 텍스트를 추출했습니다.")
        
        result = {"extracted_text": extracted_text}
        
        if progress_callback:
            progress_callback("AI 분석 준비 중...", 0.5)
        
        # OpenAI 클라이언트 설정 (API 키 검증 강화)
        api_key = request_data.get('openai_api_key')
        if api_key and api_key.strip() and not api_key.startswith('sk-') == False:
            try:
                # OpenAI 클라이언트 초기화 시 base_url 제거하여 기본 설정 사용
                client = openai.OpenAI(
                    api_key=api_key.strip()
                )
                
                # 요약 생성
                if request_data['options'].get('generate_summary'):
                    if progress_callback:
                        progress_callback("문서 요약 생성 중...", 0.7)
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "당신은 한국어 문서 요약 전문가입니다. 주요 내용을 간결하게 정리해주세요."},
                                {"role": "user", "content": f"다음 문서를 요약해주세요:\n\n{extracted_text[:3000]}"}
                            ],
                            max_tokens=500,
                            temperature=0.7
                        )
                        result['summary'] = response.choices[0].message.content
                    except Exception as e:
                        result['summary'] = f"요약 생성 중 오류: {str(e)}\n\n💡 OpenAI API 키가 올바른지 확인해주세요."
                
                # 질문-답변 생성
                if request_data['options'].get('generate_qa'):
                    if progress_callback:
                        progress_callback("질문-답변 생성 중...", 0.9)
                    
                    try:
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "문서 내용을 바탕으로 3개의 질문과 답변을 생성해주세요."},
                                {"role": "user", "content": f"다음 문서에서 중요한 질문 3개와 답변을 만들어주세요:\n\n{extracted_text[:2000]}"}
                            ],
                            max_tokens=800,
                            temperature=0.7
                        )
                        
                        qa_text = response.choices[0].message.content
                        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
                        qa_pairs = []
                        lines = qa_text.split('\n')
                        current_q = ""
                        current_a = ""
                        
                        for line in lines:
                            if line.startswith('Q') or line.startswith('질문'):
                                if current_q and current_a:
                                    qa_pairs.append({"question": current_q, "answer": current_a})
                                current_q = line
                                current_a = ""
                            elif line.startswith('A') or line.startswith('답변'):
                                current_a = line
                            elif current_a and line.strip():
                                current_a += " " + line.strip()
                        
                        if current_q and current_a:
                            qa_pairs.append({"question": current_q, "answer": current_a})
                        
                        result['qa_pairs'] = qa_pairs[:3]  # 최대 3개
                        
                    except Exception as e:
                        result['qa_pairs'] = [{"question": "질문 생성 오류", "answer": f"오류: {str(e)}"}]
            
            except Exception as e:
                result['api_error'] = f"OpenAI API 연결 오류: {str(e)}"
        
        if progress_callback:
            progress_callback("처리 완료!", 1.0)
        
        return result
        
    except Exception as e:
        return {"error": f"PDF 처리 중 오류: {str(e)}"}

# 페이지 설정 (모바일 반응형)
st.set_page_config(
    page_title="HangulPDF AI Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto"
)

# 모바일 반응형 CSS 추가
st.markdown("""
<style>
    /* 모바일 반응형 스타일 */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            font-size: 0.8rem;
            padding: 0.5rem 0.8rem;
        }
        
        .stTextArea textarea {
            font-size: 0.9rem;
        }
        
        .stButton button {
            width: 100%;
            margin-bottom: 0.5rem;
        }
        
        .stColumns {
            gap: 0.5rem;
        }
        
        .stFileUploader {
            margin-bottom: 1rem;
        }
    }
    
    /* 태블릿 반응형 */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
        }
    }
    
    /* 진행률 바 스타일 개선 */
    .stProgress > div > div > div > div {
        background-color: #ff4b4b;
    }
    
    /* 카드 스타일 */
    .info-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* 모바일에서 사이드바 자동 축소 */
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 0px;
        }
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.title("📄 HangulPDF AI Converter")
st.markdown("한글 PDF 문서를 AI가 활용하기 쉬운 형태로 변환하고 분석하는 도구입니다.")

# OCR 상태 표시
if OCR_AVAILABLE:
    st.success("🔍 OCR 기능이 활성화되었습니다. 이미지 기반 PDF도 처리할 수 있습니다.")
else:
    st.warning("⚠️ OCR 라이브러리가 설치되지 않았습니다. 텍스트 기반 PDF만 처리 가능합니다.")

# API 서버 URL (로컬 개발용)
API_BASE_URL = "http://localhost:8000"

# 사이드바 - API 키 설정
st.sidebar.header("🔑 API 설정")
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password", 
    value=st.secrets.get("OPENAI_API_KEY", ""),
    help="sk-로 시작하는 OpenAI API 키를 입력하세요"
)

if not openai_api_key:
    st.sidebar.warning("⚠️ OpenAI API 키를 입력해주세요.")
    st.sidebar.info("💡 텍스트 추출 기능은 API 키 없이도 사용 가능합니다.")

# 사이드바 - 기능 정보
st.sidebar.header("🚀 주요 기능")
st.sidebar.markdown("""
- 📄 **텍스트 추출**: PDF에서 텍스트 자동 추출
- 🔍 **OCR 지원**: 이미지 기반 PDF 처리
- 🤖 **AI 요약**: OpenAI를 활용한 문서 요약
- ❓ **Q&A 생성**: 자동 질문-답변 생성
- 📱 **모바일 지원**: 다양한 디바이스에서 사용 가능
""")

# 메인 컨텐츠
tab1, tab2, tab3 = st.tabs(["📄 PDF 업로드 & 변환", "📊 분석 결과", "🔗 공유 & 내보내기"])

with tab1:
    st.header("PDF 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=['pdf'],
        help="한글로 작성된 PDF 문서를 업로드하세요. 이미지 기반 PDF도 지원합니다."
    )
    
    if uploaded_file is not None:
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
        
        # 파일 정보 표시 (모바일 친화적)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <strong>📁 파일명:</strong> {uploaded_file.name}<br>
                <strong>📏 파일 크기:</strong> {len(uploaded_file.getvalue()):,} bytes<br>
                <strong>📋 파일 타입:</strong> {uploaded_file.type}
            </div>
            """, unsafe_allow_html=True)
        
        # 변환 옵션
        st.subheader("🔧 변환 옵션")
        
        # 모바일에서는 세로로, 데스크톱에서는 가로로 배치
        if st.session_state.get('mobile_view', False):
            extract_text = st.checkbox("📝 텍스트 추출", value=True)
            generate_summary = st.checkbox("📋 요약 생성", value=True, disabled=not openai_api_key)
            generate_qa = st.checkbox("❓ 질문-답변 생성", value=False, disabled=not openai_api_key)
            clean_text = st.checkbox("🧹 텍스트 정제", value=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                extract_text = st.checkbox("📝 텍스트 추출", value=True)
                generate_summary = st.checkbox("📋 요약 생성", value=True, disabled=not openai_api_key)
            with col2:
                generate_qa = st.checkbox("❓ 질문-답변 생성", value=False, disabled=not openai_api_key)
                clean_text = st.checkbox("🧹 텍스트 정제", value=True)
        
        # 변환 실행
        if st.button("🚀 변환 시작", type="primary", use_container_width=True):
            # 진행률 표시 컨테이너
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(text, value):
                    progress_bar.progress(value)
                    status_text.text(text)
                    time.sleep(0.1)  # 시각적 효과
                
                try:
                    # 파일을 base64로 인코딩
                    update_progress("파일 준비 중...", 0.05)
                    file_content = base64.b64encode(uploaded_file.getvalue()).decode()
                    
                    # 변환 요청 데이터
                    request_data = {
                        "file_content": file_content,
                        "filename": uploaded_file.name,
                        "options": {
                            "extract_text": extract_text,
                            "generate_summary": generate_summary,
                            "generate_qa": generate_qa,
                            "clean_text": clean_text
                        },
                        "openai_api_key": openai_api_key
                    }
                    
                    # 세션 상태에 결과 저장 (API 서버 없이 직접 처리)
                    st.session_state.conversion_result = process_pdf_locally(request_data, update_progress)
                    
                    # 완료 메시지
                    progress_bar.progress(1.0)
                    status_text.text("✅ 변환이 완료되었습니다!")
                    time.sleep(1)
                    
                    # 진행률 표시 제거
                    progress_container.empty()
                    st.success("🎉 PDF 변환이 성공적으로 완료되었습니다!")
                    st.rerun()
                    
                except Exception as e:
                    progress_container.empty()
                    st.error(f"❌ 변환 중 오류가 발생했습니다: {str(e)}")

with tab2:
    st.header("📊 분석 결과")
    
    if 'conversion_result' in st.session_state:
        result = st.session_state.conversion_result
        
        # 오류 처리
        if 'error' in result:
            st.error(f"❌ {result['error']}")
        elif 'api_error' in result:
            st.error(f"❌ {result['api_error']}")
        else:
            # 텍스트 추출 결과
            if 'extracted_text' in result:
                st.subheader("📝 추출된 텍스트")
                
                # 텍스트 길이 정보
                text_length = len(result['extracted_text'])
                st.info(f"📏 추출된 텍스트 길이: {text_length:,} 글자")
                
                with st.expander("전체 텍스트 보기"):
                    st.text_area("추출된 텍스트", value=result['extracted_text'], height=300)
            
            # 요약 결과
            if 'summary' in result:
                st.subheader("📋 문서 요약")
                st.markdown(result['summary'])
            
            # 질문-답변 결과
            if 'qa_pairs' in result:
                st.subheader("❓ 생성된 질문-답변")
                for i, qa in enumerate(result['qa_pairs'], 1):
                    with st.expander(f"Q{i}: {qa['question']}"):
                        st.write(f"**답변:** {qa['answer']}")
    else:
        st.info("📤 먼저 PDF 파일을 업로드하고 변환해주세요.")

with tab3:
    st.header("🔗 공유 & 내보내기")
    
    if 'conversion_result' in st.session_state:
        result = st.session_state.conversion_result
        
        if 'error' not in result and 'api_error' not in result:
            # 모바일에서는 세로로, 데스크톱에서는 가로로 배치
            if st.session_state.get('mobile_view', False):
                # 모바일 레이아웃
                st.subheader("📤 텍스트 내보내기")
                if 'extracted_text' in result:
                    st.download_button(
                        label="📄 텍스트 파일 다운로드",
                        data=result['extracted_text'],
                        file_name=f"extracted_text.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                if 'summary' in result:
                    st.download_button(
                        label="📋 요약 파일 다운로드",
                        data=result['summary'],
                        file_name=f"summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.subheader("🤖 AI 모델 연동")
            else:
                # 데스크톱 레이아웃
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📤 텍스트 내보내기")
                    if 'extracted_text' in result:
                        st.download_button(
                            label="📄 텍스트 파일 다운로드",
                            data=result['extracted_text'],
                            file_name=f"extracted_text.txt",
                            mime="text/plain"
                        )
                    
                    if 'summary' in result:
                        st.download_button(
                            label="📋 요약 파일 다운로드",
                            data=result['summary'],
                            file_name=f"summary.txt",
                            mime="text/plain"
                        )
                
                with col2:
                    st.subheader("🤖 AI 모델 연동")
            
            if 'extracted_text' in result:
                # ChatGPT 프롬프트 (전체 텍스트 표시)
                st.markdown("**💬 ChatGPT 프롬프트:**")
                chatgpt_prompt = f"""다음 한글 문서를 AI가 자동 분석한 뒤, 문서 유형과 주요 내용을 파악하여 다음 항목들을 포함한 요약 및 구조화된 분석 결과를 생성해주세요.

{result['extracted_text']}

1. 📂 문서 기본 정보:
   - 문서 제목 또는 추정 제목
   - 작성 날짜 또는 추정 시점
   - 작성 주체 또는 관련 기관/담당자 추정
   - 문서 목적(정책 문서/보고서/계획안/회의록/제안서 등) 자동 분류

2. 🧩 문서 구조 분석:
   - 목차 또는 섹션 구성 추정
   - 각 섹션별 요약 (3줄 이내)
   - 표, 그림, 도표가 포함된 경우 해당 내용 요약

3. 🧠 핵심 내용 요약 및 인사이트:
   - 전체 문서의 핵심 주제 및 주요 주장 요약 (5줄 이내)
   - 자주 등장하는 키워드 및 핵심 개념(빈도 분석 포함)
   - 문서 내 등장하는 중요한 수치, 날짜, 고유명사(인물, 기관 등) 추출
   - 중요한 결정사항, 요청사항, 일정, 액션 아이템 자동 분리

4. 🛠️ 문서 유형별 특화 분석 (자동 판단하여 포함):
   - ✅ 기획안/제안서: 핵심 아이디어, 제안 배경, 기대 효과 요약
   - ✅ 회의록: 참석자, 주요 논의사항, 결정사항 및 후속 조치 정리
   - ✅ 정책/행정문서: 정책 목적, 대상, 추진 전략 및 일정 요약
   - ✅ 공사/계약문서: 계약 조건, 공정 일정, 이해관계자 분석
   - ✅ 보고서: 분석 대상, 방법, 결론 및 제언 구분

5. 🔍 오류 및 주의요소 감지:
   - 문서 내 날짜 오류, 논리 비약, 누락 정보 자동 감지
   - 문맥상 혼란을 줄 수 있는 표현 또는 오탈자 추정

6. 🧾 결과 요약 형식:
   - 마크다운(.md) 형식으로 요약 결과 제공
   - 제목, 소제목, 목록 등을 구조적으로 제공

문서를 사람이 읽지 않고도 전체적 흐름과 인사이트를 파악할 수 있도록 분석해주세요."""
                
                st.text_area(
                    "ChatGPT에 복사하여 사용하세요:", 
                    value=chatgpt_prompt, 
                    height=200,
                    key="chatgpt_prompt"
                )
                
                # Gemini 프롬프트
                st.markdown("**🔮 Gemini 프롬프트:**")
                gemini_prompt = f"""다음 한글 문서를 AI가 자동 분석한 뒤, 문서 유형과 주요 내용을 파악하여 다음 항목들을 포함한 요약 및 구조화된 분석 결과를 생성해주세요.

{result['extracted_text']}

1. 📂 문서 기본 정보:
   - 문서 제목 또는 추정 제목
   - 작성 날짜 또는 추정 시점
   - 작성 주체 또는 관련 기관/담당자 추정
   - 문서 목적(정책 문서/보고서/계획안/회의록/제안서 등) 자동 분류

2. 🧩 문서 구조 분석:
   - 목차 또는 섹션 구성 추정
   - 각 섹션별 요약 (3줄 이내)
   - 표, 그림, 도표가 포함된 경우 해당 내용 요약

3. 🧠 핵심 내용 요약 및 인사이트:
   - 전체 문서의 핵심 주제 및 주요 주장 요약 (5줄 이내)
   - 자주 등장하는 키워드 및 핵심 개념(빈도 분석 포함)
   - 문서 내 등장하는 중요한 수치, 날짜, 고유명사(인물, 기관 등) 추출
   - 중요한 결정사항, 요청사항, 일정, 액션 아이템 자동 분리

4. 🛠️ 문서 유형별 특화 분석 (자동 판단하여 포함):
   - ✅ 기획안/제안서: 핵심 아이디어, 제안 배경, 기대 효과 요약
   - ✅ 회의록: 참석자, 주요 논의사항, 결정사항 및 후속 조치 정리
   - ✅ 정책/행정문서: 정책 목적, 대상, 추진 전략 및 일정 요약
   - ✅ 공사/계약문서: 계약 조건, 공정 일정, 이해관계자 분석
   - ✅ 보고서: 분석 대상, 방법, 결론 및 제언 구분

5. 🔍 오류 및 주의요소 감지:
   - 문서 내 날짜 오류, 논리 비약, 누락 정보 자동 감지
   - 문맥상 혼란을 줄 수 있는 표현 또는 오탈자 추정

6. 🧾 결과 요약 형식:
   - 마크다운(.md) 형식으로 요약 결과 제공
   - 제목, 소제목, 목록 등을 구조적으로 제공

문서를 사람이 읽지 않고도 전체적 흐름과 인사이트를 파악할 수 있도록 분석해주세요."""
                
                st.text_area(
                    "Gemini에 복사하여 사용하세요:", 
                    value=gemini_prompt, 
                    height=150,
                    key="gemini_prompt"
                )
                
                # Grok 프롬프트
                st.markdown("**🚀 Grok 프롬프트:**")
                grok_prompt = f"""Analyze the uploaded Korean-language PDF file and provide the following structured output in Markdown format in Korean:

{result['extracted_text']}

- Document type, estimated title, date, and author/institution
- Automatic detection of document structure (sections, tables, etc.)
- Summary of each section (up to 3 lines)
- Extraction of key entities (names, organizations, numbers, dates)
- Top keywords by frequency
- Key insights or action items if applicable
- Special treatment based on document type (proposal, report, minutes, etc.)
- Highlight any inconsistencies, logical errors, or missing sections"""
                
                st.text_area(
                    "Grok에 복사하여 사용하세요:", 
                    value=grok_prompt, 
                    height=150,
                    key="grok_prompt"
                )
            
        else:
            st.error("변환 결과에 오류가 있어 내보내기를 할 수 없습니다.")
    else:
        st.info("📤 먼저 PDF 파일을 업로드하고 변환해주세요.")

# 푸터
st.markdown("---")
st.markdown("🔧 **HangulPDF AI Converter** | 한글 PDF 문서 AI 변환 도구")
st.markdown("📱 모바일, 태블릿, 데스크톱 모든 디바이스에서 사용 가능합니다.")

