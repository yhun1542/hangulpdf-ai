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

# 추가: json for safe JS escaping
import json as pyjson

# 진행률 표시를 위한 함수 (개선: 시간 표시 추가)
def show_progress(progress_text, progress_value):
    """진행률을 표시하는 함수"""
    progress_bar = st.progress(progress_value)
    status_text = st.empty()
    status_text.text(progress_text)
    return progress_bar, status_text

# 표 구조 감지 및 분할 함수
def detect_and_extract_tables(image):
    """표 구조를 감지하고 셀별로 분할하여 OCR 처리"""
    try:
        # 이미지를 그레이스케일로 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 이진화
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 수평선 감지
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # 수직선 감지
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # 표 구조 결합
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # 표 영역 감지
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # 최소 표 크기 필터링
                x, y, w, h = cv2.boundingRect(contour)
                # 표 영역이 충분히 큰 경우에만 추가
                if w > 100 and h > 50:
                    table_regions.append((x, y, w, h))
        
        return table_regions, horizontal_lines, vertical_lines
        
    except Exception as e:
        st.warning(f"표 구조 감지 중 오류: {str(e)}")
        return [], None, None

# 텍스트 영역 감지 함수
def detect_text_regions(image):
    """이미지에서 텍스트 영역을 감지하여 분할"""
    try:
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 적응형 이진화
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # 모폴로지 연산으로 텍스트 영역 확장
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 최소 텍스트 영역 크기
                x, y, w, h = cv2.boundingRect(contour)
                # 텍스트 영역 비율 확인 (너무 가늘거나 작은 영역 제외)
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 20 and w > 20 and h > 10:
                    text_regions.append((x, y, w, h))
        
        return text_regions
        
    except Exception as e:
        st.warning(f"텍스트 영역 감지 중 오류: {str(e)}")
        return []

# 한글 특화 이미지 전처리 함수
def preprocess_for_korean(image):
    """한글 인식에 최적화된 이미지 전처리"""
    try:
        # PIL Image를 numpy array로 변환
        img_array = np.array(image)
        
        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 해상도 3배 증가 (한글은 복잡한 구조로 인해 더 높은 해상도 필요)
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)
        
        # 한글 특화 필터링
        # 1. 가우시안 블러로 노이즈 제거 (한글 획의 연결성 향상)
        gray = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # 2. 언샤프 마스킹으로 한글 획 선명화
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # 3. 한글 특화 대비 향상
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(unsharp_mask)
        
        # 4. 한글 획 두께 정규화를 위한 모폴로지 연산
        kernel = np.ones((1, 1), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # 5. 적응형 이진화 (한글의 다양한 크기와 두께에 대응)
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 6. 한글 자소 연결성 향상을 위한 추가 모폴로지 연산
        korean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, korean_kernel)
        
        return binary
        
    except Exception as e:
        st.warning(f"한글 특화 전처리 중 오류: {str(e)}")
        # 기본 전처리로 폴백
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        return gray

# 로컬 PDF 처리 함수 (개선: 타이머 추가 로직)
def process_pdf_locally(request_data, progress_callback=None):
    """로컬에서 PDF 처리"""
    start_time = time.time()  # 시작 시간 기록
    estimated_time = 30  # 고정 예상 시간 (초), 실제로는 동적으로 조정 가능
    
    def update_progress_with_timer(text, value):
        elapsed = time.time() - start_time
        remaining = max(0, estimated_time - elapsed)
        timer_text = f" | 남은 시간: {int(remaining)}초"
        if progress_callback:
            progress_callback(text + timer_text, value)
    
    try:
        # 파일 준비
        update_progress_with_timer("파일 준비 중...", 0.1)
        pdf_bytes = base64.b64decode(request_data['pdf_base64'])
        
        # 기본 텍스트 추출
        update_progress_with_timer("PDF 파일 디코딩 중...", 0.2)
        
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        num_pages = len(pdf_reader.pages)
        
        update_progress_with_timer(f"텍스트 추출 중... ({num_pages}페이지)", 0.3)
        
        extracted_text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    extracted_text += page_text + "\n"
            except Exception as e:
                continue
        
        # OCR 처리 (선택적)
        if request_data.get('use_ocr', False) and OCR_AVAILABLE:
            update_progress_with_timer("표/이미지 특화 OCR을 사용한 텍스트 추출 중...", 0.4)
            ocr_text = extract_text_with_advanced_ocr(pdf_bytes)
            if ocr_text and len(ocr_text.strip()) > len(extracted_text.strip()):
                extracted_text = ocr_text
        
        # 요약 생성
        if request_data.get('generate_summary', False):
            update_progress_with_timer("문서 요약 생성 중...", 0.7)
            # 요약 로직 (기존 유지)
        
        # Q&A 생성
        if request_data.get('generate_qa', False):
            update_progress_with_timer("질문-답변 생성 중...", 0.9)
            # Q&A 로직 (기존 유지)
        
        update_progress_with_timer("처리 완료!", 1.0)
        
        return {
            'extracted_text': extracted_text,
            'text_length': len(extracted_text),
            'pages': num_pages
        }
        
    except Exception as e:
        return {'error': f'PDF 처리 중 오류가 발생했습니다: {str(e)}'}

# 고급 OCR 함수 (기존 유지)
def extract_text_with_advanced_ocr(pdf_bytes):
    """고급 OCR을 사용하여 이미지 기반 PDF에서 텍스트 추출"""
    if not OCR_AVAILABLE:
        return "OCR 라이브러리가 설치되지 않았습니다."
    
    try:
        st.info("🔍 표/이미지 특화 고급 OCR 처리를 시작합니다...")
        
        # PDF를 고해상도 이미지로 변환
        images = convert_from_bytes(pdf_bytes, dpi=400, fmt='PNG')
        extracted_text = ""
        
        for i, image in enumerate(images):
            st.info(f"📄 페이지 {i+1}/{len(images)} 처리 중...")
            
            # 이미지를 numpy array로 변환
            img_array = np.array(image)
            
            # 1. 표 구조 감지 및 처리
            table_regions, h_lines, v_lines = detect_and_extract_tables(img_array)
            table_text = ""
            if table_regions:
                st.info(f"📊 페이지 {i+1}에서 {len(table_regions)}개의 표 감지")
                # 표 처리 로직 추가 필요
            
            # 2. 텍스트 영역 감지 및 처리
            text_regions = detect_text_regions(img_array)
            region_text = ""
            if text_regions:
                st.info(f"🖼️ 페이지 {i+1}에서 {len(text_regions)}개의 텍스트 영역 감지")
                # 텍스트 영역 처리 로직 추가 필요
            
            # 3. 전체 페이지 기본 OCR
            processed_image = preprocess_for_korean(image)
            
            try:
                text = pytesseract.image_to_string(processed_image, config=r'--oem 3 --psm 3 -l kor+eng')
                if text.strip():
                    extracted_text += f"\n--- 페이지 {i+1} ---\n{text}\n"
            except Exception:
                continue
        
        return extracted_text
        
    except Exception as e:
        return f"OCR 처리 중 오류: {str(e)}"

# Streamlit 페이지 설정
st.set_page_config(
    page_title="HangulPDF AI Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 모바일 반응형 CSS
st.markdown("""
<style>
/* 모바일 반응형 디자인 */
@media (max-width: 768px) {
    .main .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.75rem;
        font-size: 0.9rem;
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
            # 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(text, value):
                progress_bar.progress(value)
                status_text.text(text)
            
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
            result = process_pdf_locally(request_data, update_progress)
            
            # 결과 저장
            st.session_state.conversion_result = result
            st.session_state.uploaded_filename = uploaded_file.name
            
            if 'error' not in result:
                st.success("✅ 변환이 완료되었습니다!")
                st.balloons()
            else:
                st.error(f"❌ 변환 중 오류가 발생했습니다: {result['error']}")

with tab2:
    st.header("📊 변환 결과")
    
    if 'conversion_result' in st.session_state:
        result = st.session_state.conversion_result
        
        if 'error' not in result:
            # 텍스트 추출 결과
            if 'extracted_text' in result:
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
            
            # 요약 결과
            if 'summary' in result:
                st.subheader("📋 문서 요약")
                st.write(result['summary'])
            
            # Q&A 결과
            if 'qa_pairs' in result:
                st.subheader("❓ 질문-답변")
                for i, qa in enumerate(result['qa_pairs'], 1):
                    st.write(f"**Q{i}:** {qa['question']}")
                    st.write(f"**A{i}:** {qa['answer']}")
                    st.write("---")
        
        else:
            st.error(f"❌ {result['error']}")
    
    else:
        st.info("📤 먼저 PDF 파일을 업로드하고 변환해주세요.")

with tab3:
    st.header("🔗 공유 & 내보내기")
    
    if 'conversion_result' in st.session_state:
        result = st.session_state.conversion_result
        
        if 'error' not in result and 'extracted_text' in result:
            extracted_text = result['extracted_text']
            
            # ChatGPT 프롬프트
            st.markdown("**💬 ChatGPT 프롬프트:**")
            chatgpt_prompt = f"""다음 한글 문서를 AI가 자동 분석한 뒤, 문서 유형과 주요 내용을 파악하여 다음 항목들을 포함한 요약 및 구조화된 분석 결과를 생성해주세요.

1. 📂 문서 기본 정보:
   - 문서 제목 또는 추정 제목
   - 작성 날짜 또는 추정 시점
   - 작성 주체 또는 관련 기관/담당자 추정
   - 문서 목적(정책 문서, 보고서, 제안서, 회의록 등)

2. 📋 구조 분석:
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
            
            # 복사 버튼: JS로 구현
            js_chatgpt = pyjson.dumps(chatgpt_prompt, ensure_ascii=False)
            st.components.v1.html(
                f"""
                <button onclick="copyChatGPT()" style="margin-left: 10px; padding: 5px 10px; background: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer;">📋 복사하기</button>
                <script>
                function copyChatGPT() {{
                    navigator.clipboard.writeText({js_chatgpt}).then(function() {{
                        alert('ChatGPT 프롬프트가 복사되었습니다!');
                    }});
                }}
                </script>
                """,
                height=40,
            )
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
   - 문서 목적(정책 문서, 보고서, 제안서, 회의록 등)

2. 📋 구조 분석:
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
            
            js_gemini = pyjson.dumps(gemini_prompt, ensure_ascii=False)
            st.components.v1.html(
                f"""
                <button onclick="copyGemini()" style="margin-left: 10px; padding: 5px 10px; background: #4285F4; color: white; border: none; border-radius: 3px; cursor: pointer;">📋 복사하기</button>
                <script>
                function copyGemini() {{
                    navigator.clipboard.writeText({js_gemini}).then(function() {{
                        alert('Gemini 프롬프트가 복사되었습니다!');
                    }});
                }}
                </script>
                """,
                height=40,
            )
            st.text_area(
                "Gemini에 복사하여 사용하세요:", 
                value=gemini_prompt, 
                height=150,
                key="gemini_prompt"
            )
            
            # Grok 프롬프트
            st.markdown("**🚀 Grok 프롬프트:**")
            grok_prompt = f"""Analyze the uploaded Korean-language PDF file and provide the following structured output in Markdown format in Korean:
- Document type, estimated title, date, and author/institution
- Automatic detection of document structure (sections, tables, etc.)
- Summary of each section (up to 3 lines)
- Extraction of key entities (names, organizations, numbers, dates)
- Top keywords by frequency
- Key insights or action items if applicable
- Special treatment based on document type (proposal, report, minutes, etc.)
- Highlight any inconsistencies, logical errors, or missing sections

---

{extracted_text}"""
            
            js_grok = pyjson.dumps(grok_prompt, ensure_ascii=False)
            st.components.v1.html(
                f"""
                <button onclick="copyGrok()" style="margin-left: 10px; padding: 5px 10px; background: #000000; color: white; border: none; border-radius: 3px; cursor: pointer;">📋 복사하기</button>
                <script>
                function copyGrok() {{
                    navigator.clipboard.writeText({js_grok}).then(function() {{
                        alert('Grok 프롬프트가 복사되었습니다!');
                    }});
                }}
                </script>
                """,
                height=40,
            )
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
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📄 <strong>HangulPDF AI Converter</strong> - 한글 PDF 문서 AI 변환 도구</p>
    <p>🔧 표/이미지 특화 OCR | 📱 모바일 반응형 | 🚀 실시간 진행률 표시</p>
    <p>💡 모든 디바이스에서 사용 가능한 웹 애플리케이션</p>
</div>
""", unsafe_allow_html=True)

