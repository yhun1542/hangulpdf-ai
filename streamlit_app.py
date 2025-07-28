import streamlit as st
import requests
import json
import os
from io import BytesIO
import base64
import time
import re

# OCR 및 이미지 처리를 위한 라이브러리
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# PDF 처리를 위한 라이브러리
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# 진행률 표시를 위한 함수 (타이머 수정)
def show_progress_with_timer(progress_text, progress_value, start_time, estimated_time=30):
    """진행률과 타이머를 표시하는 함수"""
    elapsed = time.time() - start_time
    remaining = max(0, estimated_time - elapsed)
    timer_text = f" | 남은 시간: {int(remaining)}초"
    
    progress_bar = st.progress(progress_value)
    status_text = st.empty()
    status_text.text(progress_text + timer_text)
    return progress_bar, status_text

# 로컬 PDF 처리 함수 (수정: 안정성 향상)
def process_pdf_locally(request_data):
    """로컬에서 PDF 처리 (안정성 향상)"""
    start_time = time.time()
    estimated_time = 30
    
    try:
        # 1. 파일 준비
        progress_bar, status_text = show_progress_with_timer("파일 준비 중...", 0.1, start_time, estimated_time)
        time.sleep(0.5)  # UI 업데이트를 위한 짧은 대기
        
        if not request_data.get('pdf_base64'):
            return {'error': 'PDF 데이터가 없습니다.'}
        
        pdf_bytes = base64.b64decode(request_data['pdf_base64'])
        
        # 2. PDF 디코딩
        progress_bar, status_text = show_progress_with_timer("PDF 파일 디코딩 중...", 0.2, start_time, estimated_time)
        time.sleep(0.5)
        
        if not PDF_AVAILABLE:
            return {'error': 'PyPDF2 라이브러리가 설치되지 않았습니다.'}
        
        # 3. 기본 텍스트 추출
        progress_bar, status_text = show_progress_with_timer("텍스트 추출 중...", 0.3, start_time, estimated_time)
        time.sleep(0.5)
        
        extracted_text = ""
        num_pages = 0
        
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            num_pages = len(pdf_reader.pages)
            
            st.info(f"📄 PDF 페이지 수: {num_pages}")
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        extracted_text += f"\n--- 페이지 {page_num + 1} ---\n"
                        extracted_text += page_text + "\n"
                        st.success(f"✅ 페이지 {page_num + 1} 텍스트 추출 성공")
                    else:
                        st.warning(f"⚠️ 페이지 {page_num + 1} 텍스트 추출 실패 (빈 페이지 또는 이미지 기반)")
                except Exception as e:
                    st.warning(f"⚠️ 페이지 {page_num + 1} 처리 중 오류: {str(e)}")
                    continue
                    
        except Exception as e:
            st.error(f"❌ PDF 읽기 오류: {str(e)}")
            return {'error': f'PDF 읽기 실패: {str(e)}'}
        
        # 4. OCR 처리 (선택적)
        if request_data.get('use_ocr', False) and OCR_AVAILABLE:
            progress_bar, status_text = show_progress_with_timer("OCR을 사용한 텍스트 추출 중...", 0.5, start_time, estimated_time)
            time.sleep(0.5)
            
            try:
                ocr_text = extract_text_with_basic_ocr(pdf_bytes)
                if ocr_text and len(ocr_text.strip()) > len(extracted_text.strip()):
                    extracted_text = ocr_text
                    st.success("✅ OCR 텍스트 추출 완료")
                elif ocr_text:
                    extracted_text += f"\n=== OCR 추가 텍스트 ===\n{ocr_text}"
                    st.info("ℹ️ OCR 텍스트를 추가로 결합했습니다")
            except Exception as e:
                st.warning(f"⚠️ OCR 처리 중 오류: {str(e)}")
        
        # 5. 결과 검증
        progress_bar, status_text = show_progress_with_timer("결과 검증 중...", 0.8, start_time, estimated_time)
        time.sleep(0.5)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            return {
                'error': '텍스트 추출에 실패했습니다. OCR 옵션을 사용해보세요.',
                'extracted_text': extracted_text,
                'text_length': len(extracted_text),
                'pages': num_pages
            }
        
        # 6. 완료
        progress_bar, status_text = show_progress_with_timer("처리 완료!", 1.0, start_time, estimated_time)
        time.sleep(0.5)
        
        st.success(f"✅ 텍스트 추출 완료: {len(extracted_text)} 글자")
        
        return {
            'extracted_text': extracted_text,
            'text_length': len(extracted_text),
            'pages': num_pages,
            'success': True
        }
        
    except Exception as e:
        st.error(f"❌ 처리 중 예상치 못한 오류: {str(e)}")
        return {'error': f'PDF 처리 중 오류가 발생했습니다: {str(e)}'}

# 기본 OCR 처리 함수 (간소화)
def extract_text_with_basic_ocr(pdf_bytes):
    """기본 OCR을 사용하여 텍스트 추출"""
    if not OCR_AVAILABLE:
        return "OCR 라이브러리가 설치되지 않았습니다."
    
    try:
        st.info("🔍 OCR 처리를 시작합니다...")
        
        # PDF를 이미지로 변환
        images = convert_from_bytes(pdf_bytes, dpi=300, fmt='PNG')
        extracted_text = ""
        
        for i, image in enumerate(images):
            st.info(f"📄 페이지 {i+1}/{len(images)} OCR 처리 중...")
            
            try:
                # 기본 OCR 설정
                config = r'--oem 3 --psm 3 -l kor+eng'
                text = pytesseract.image_to_string(image, config=config)
                
                if text.strip():
                    extracted_text += f"\n--- 페이지 {i+1} (OCR) ---\n"
                    extracted_text += text + "\n"
                    st.success(f"✅ 페이지 {i+1} OCR 완료")
                else:
                    st.warning(f"⚠️ 페이지 {i+1} OCR 결과 없음")
                    
            except Exception as e:
                st.warning(f"⚠️ 페이지 {i+1} OCR 처리 중 오류: {str(e)}")
                continue
        
        return extracted_text
        
    except Exception as e:
        st.error(f"❌ OCR 처리 중 오류: {str(e)}")
        return f"OCR 처리 중 오류: {str(e)}"

# Streamlit 페이지 설정
st.set_page_config(
    page_title="HangulPDF AI Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
/* 모바일 반응형 디자인 */
@media (max-width: 768px) {
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

/* 정보 카드 스타일 */
.info-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

/* 버튼 스타일 */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 5px;
    padding: 0.5rem 1rem;
    transition: transform 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
}

/* 진행률 바 스타일 */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
}
</style>
""", unsafe_allow_html=True)

# 메인 제목
st.title("📄 HangulPDF AI Converter")
st.markdown("**한글 PDF 문서를 AI가 쉽게 활용할 수 있도록 자동 변환하는 도구**")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # API 키 입력
    api_key = st.text_input("🔑 OpenAI API 키", type="password", help="GPT 기반 요약 및 Q&A 생성에 필요합니다.")
    
    st.header("🔧 변환 옵션")
    
    # 변환 옵션들
    extract_text = st.checkbox("📝 텍스트 추출", value=True, help="PDF에서 텍스트를 추출합니다.")
    
    use_ocr = st.checkbox(
        "🔍 OCR 사용", 
        value=False, 
        disabled=not OCR_AVAILABLE,
        help="이미지 기반 PDF나 스캔된 문서에서 텍스트를 추출합니다. 처리 시간이 더 오래 걸릴 수 있습니다."
    )
    
    if not OCR_AVAILABLE:
        st.warning("⚠️ OCR 라이브러리가 설치되지 않았습니다.")
    
    if not PDF_AVAILABLE:
        st.error("❌ PyPDF2 라이브러리가 설치되지 않았습니다.")
    
    generate_summary = st.checkbox("📋 요약 생성", value=False, help="OpenAI API를 사용하여 문서 요약을 생성합니다.")
    generate_qa = st.checkbox("❓ 질문-답변 생성", value=False, help="문서 내용 기반 질문과 답변을 생성합니다.")
    
    if (generate_summary or generate_qa) and not api_key:
        st.warning("⚠️ API 키가 필요합니다.")

# 메인 탭
tab1, tab2, tab3 = st.tabs(["📤 파일 업로드", "📊 변환 결과", "🔗 공유 & 내보내기"])

with tab1:
    st.header("📤 PDF 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=['pdf'],
        help="최대 200MB까지 업로드 가능합니다."
    )
    
    if uploaded_file is not None:
        # 파일 정보 표시
        file_size = len(uploaded_file.getvalue())
        
        st.markdown(f"""
        <div class="info-card">
            <h4>📁 업로드된 파일 정보</h4>
            <p><strong>📁 파일명:</strong> {uploaded_file.name}</p>
            <p><strong>📏 파일 크기:</strong> {file_size:,} bytes</p>
            <p><strong>📋 파일 타입:</strong> {uploaded_file.type}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # OCR 모드 안내
        if use_ocr:
            st.info("🔍 OCR 모드: 이미지 기반 PDF에서도 텍스트를 추출합니다. 처리 시간이 더 오래 걸릴 수 있습니다.")
        else:
            st.info("📄 기본 모드: 텍스트 기반 PDF에서만 텍스트를 추출합니다. 빠른 처리가 가능합니다.")
        
        # 변환 버튼
        if st.button("🚀 변환 시작", type="primary"):
            if not PDF_AVAILABLE:
                st.error("❌ PyPDF2 라이브러리가 설치되지 않았습니다. 관리자에게 문의하세요.")
            else:
                # PDF 처리
                pdf_bytes = uploaded_file.getvalue()
                pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
                
                request_data = {
                    'pdf_base64': pdf_base64,
                    'extract_text': extract_text,
                    'use_ocr': use_ocr,
                    'generate_summary': generate_summary,
                    'generate_qa': generate_qa,
                    'api_key': api_key
                }
                
                # 로컬 처리
                result = process_pdf_locally(request_data)
                
                # 결과 저장
                st.session_state.conversion_result = result
                st.session_state.uploaded_filename = uploaded_file.name
                
                if 'error' not in result or result.get('success'):
                    st.success("✅ 변환이 완료되었습니다!")
                    st.balloons()
                else:
                    st.error(f"❌ 변환 중 오류가 발생했습니다: {result.get('error', '알 수 없는 오류')}")

with tab2:
    st.header("📊 변환 결과")
    
    if 'conversion_result' in st.session_state:
        result = st.session_state.conversion_result
        
        if 'extracted_text' in result and result.get('extracted_text'):
            # 텍스트 추출 결과
            st.subheader("📝 추출된 텍스트")
            
            # 텍스트 정보
            text_length = result.get('text_length', 0)
            pages = result.get('pages', 0)
            
            st.markdown(f"""
            <div class="info-card">
                <h4>📊 텍스트 정보</h4>
                <p><strong>📏 텍스트 길이:</strong> {text_length:,} 글자</p>
                <p><strong>📄 페이지 수:</strong> {pages} 페이지</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 텍스트 표시
            st.text_area(
                "추출된 텍스트:",
                value=result['extracted_text'],
                height=400,
                key="extracted_text_display"
            )
        
        elif 'error' in result:
            st.error(f"❌ {result['error']}")
            if 'extracted_text' in result:
                st.info("부분적으로 추출된 텍스트:")
                st.text_area("부분 텍스트:", value=result['extracted_text'], height=200)
        
        else:
            st.warning("⚠️ 추출된 텍스트가 없습니다.")
    
    else:
        st.info("📤 먼저 PDF 파일을 업로드하고 변환해주세요.")

with tab3:
    st.header("🔗 공유 & 내보내기")
    
    if 'conversion_result' in st.session_state:
        result = st.session_state.conversion_result
        
        if 'extracted_text' in result and result.get('extracted_text'):
            extracted_text = result['extracted_text']
            
            # ChatGPT 프롬프트
            st.markdown("**💬 ChatGPT 프롬프트:**")
            chatgpt_prompt = f"""다음 한글 문서를 AI가 자동 분석한 뒤, 문서 유형과 주요 내용을 파악하여 다음 항목들을 포함한 요약 및 구조화된 분석 결과를 생성해주세요.

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

문서를 사람이 읽지 않고도 전체적 흐름과 인사이트를 파악할 수 있도록 분석해주세요.

---

{extracted_text}"""
            
            st.text_area(
                "ChatGPT에 복사하여 사용하세요:", 
                value=chatgpt_prompt, 
                height=200,
                key="chatgpt_prompt"
            )
            
            # Gemini 프롬프트
            st.markdown("**🔮 Gemini 프롬프트:**")
            gemini_prompt = f"""다음 한글 문서를 AI가 자동 분석한 뒤, 문서 유형과 주요 내용을 파악하여 다음 항목들을 포함한 요약 및 구조화된 분석 결과를 생성해주세요.

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

문서를 사람이 읽지 않고도 전체적 흐름과 인사이트를 파악할 수 있도록 분석해주세요.

---

{extracted_text}"""
            
            st.text_area(
                "Gemini에 복사하여 사용하세요:", 
                value=gemini_prompt, 
                height=150,
                key="gemini_prompt"
            )
            
            # Grok 프롬프트
            st.markdown("**🚀 Grok 프롬프트:**")
            grok_prompt = f"""Analyze this Korean document and provide insights in Korean:

- Document type and key themes
- Important data points and statistics
- Main conclusions and recommendations
- Creative perspectives on the content

---

{extracted_text}"""
            
            st.text_area(
                "Grok에 복사하여 사용하세요:", 
                value=grok_prompt, 
                height=150,
                key="grok_prompt"
            )
            
        else:
            st.error("변환 결과에 텍스트가 없어 내보내기를 할 수 없습니다.")
    else:
        st.info("📤 먼저 PDF 파일을 업로드하고 변환해주세요.")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📄 <strong>HangulPDF AI Converter</strong> - 한글 PDF 문서 AI 변환 도구</p>
    <p>🔧 안정성 향상 | 📱 모바일 반응형 | ⏱️ 실시간 타이머</p>
    <p>💡 모든 디바이스에서 사용 가능한 웹 애플리케이션</p>
</div>
""", unsafe_allow_html=True)

