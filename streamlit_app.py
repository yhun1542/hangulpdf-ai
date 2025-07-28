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

# 진행률 표시를 위한 함수
def show_progress(progress_text, progress_value):
    """진행률을 표시하는 함수"""
    progress_bar = st.progress(progress_value)
    status_text = st.empty()
    status_text.text(progress_text)
    return progress_bar, status_text

# 고급 이미지 전처리 함수
def preprocess_image_advanced(image):
    """고급 이미지 전처리로 OCR 정확도 향상"""
    try:
        # PIL Image를 numpy array로 변환
        img_array = np.array(image)
        
        # 1. 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # 2. 해상도 증가 (2배 확대)
        height, width = gray.shape
        gray = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
        
        # 3. 가우시안 블러로 노이즈 제거
        gray = cv2.GaussianBlur(gray, (1, 1), 0)
        
        # 4. 대비 향상 (CLAHE - Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 5. 모폴로지 연산으로 텍스트 선명화
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 6. 이진화 (Otsu's thresholding)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 7. 기울기 보정 (Skew correction)
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # 기울기가 1도 이상일 때만 보정
            if abs(angle) > 1:
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return binary
        
    except Exception as e:
        st.warning(f"이미지 전처리 중 오류: {str(e)}")
        # 기본 전처리로 폴백
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        return gray

# 텍스트 후처리 함수
def postprocess_text(text):
    """OCR 결과 텍스트 후처리로 정확도 향상"""
    if not text:
        return text
    
    # 1. 일반적인 OCR 오류 수정
    corrections = {
        # 한글 자주 틀리는 글자들
        'ㅇ': 'o',  # 영문 o와 한글 ㅇ 구분
        'ㅁ': 'm',  # 영문 m과 한글 ㅁ 구분
        '|': 'l',   # 세로선과 영문 l 구분
        '0': 'O',   # 숫자 0과 영문 O 구분 (문맥에 따라)
        '1': 'l',   # 숫자 1과 영문 l 구분 (문맥에 따라)
    }
    
    # 2. 불필요한 공백 정리
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백을 하나로
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 연속된 줄바꿈 정리
    
    # 3. 한글 문장 부호 정리
    text = re.sub(r'["""]', '"', text)  # 따옴표 통일
    text = re.sub(r'[''']', "'", text)  # 작은따옴표 통일
    text = re.sub(r'[…]', '...', text)  # 말줄임표 통일
    
    # 4. 숫자와 단위 사이 공백 정리
    text = re.sub(r'(\d)\s+([가-힣]{1,2})', r'\1\2', text)  # "10 개" -> "10개"
    text = re.sub(r'(\d)\s+(%|원|달러|kg|m|cm)', r'\1\2', text)  # "100 %" -> "100%"
    
    # 5. 날짜 형식 정리
    text = re.sub(r'(\d{4})\s*[년]\s*(\d{1,2})\s*[월]\s*(\d{1,2})\s*[일]', r'\1년 \2월 \3일', text)
    
    return text.strip()

# 다단계 OCR 처리 함수
def extract_text_with_advanced_ocr(pdf_bytes):
    """고급 OCR을 사용하여 이미지 기반 PDF에서 텍스트 추출 (정확도 향상)"""
    if not OCR_AVAILABLE:
        return "OCR 라이브러리가 설치되지 않았습니다."
    
    try:
        st.info("🔍 고급 OCR 처리를 시작합니다...")
        
        # PDF를 고해상도 이미지로 변환
        images = convert_from_bytes(pdf_bytes, dpi=400, fmt='PNG')  # DPI 증가
        extracted_text = ""
        
        # 다양한 OCR 설정들
        ocr_configs = [
            # 설정 1: 한국어 + 영어, 자동 페이지 분할
            {
                'config': r'--oem 3 --psm 3 -l kor+eng',
                'name': '자동 페이지 분할'
            },
            # 설정 2: 한국어 + 영어, 단일 텍스트 블록
            {
                'config': r'--oem 3 --psm 6 -l kor+eng',
                'name': '단일 텍스트 블록'
            },
            # 설정 3: 한국어 + 영어, 단일 텍스트 라인
            {
                'config': r'--oem 3 --psm 7 -l kor+eng',
                'name': '단일 텍스트 라인'
            },
            # 설정 4: 한국어 + 영어, 단일 단어
            {
                'config': r'--oem 3 --psm 8 -l kor+eng',
                'name': '단일 단어'
            }
        ]
        
        for i, image in enumerate(images):
            st.info(f"📄 페이지 {i+1}/{len(images)} 처리 중...")
            
            # 고급 이미지 전처리
            processed_image = preprocess_image_advanced(image)
            
            best_text = ""
            best_confidence = 0
            
            # 여러 OCR 설정으로 시도
            for config_info in ocr_configs:
                try:
                    # OCR 실행
                    text = pytesseract.image_to_string(processed_image, config=config_info['config'])
                    
                    # 신뢰도 측정 (텍스트 길이와 한글 비율로 추정)
                    if text.strip():
                        korean_chars = len(re.findall(r'[가-힣]', text))
                        total_chars = len(re.sub(r'\s', '', text))
                        
                        if total_chars > 0:
                            confidence = (korean_chars / total_chars) * len(text.strip())
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_text = text
                                st.success(f"✅ 최적 설정: {config_info['name']} (신뢰도: {confidence:.1f})")
                
                except Exception as e:
                    continue
            
            # 최적 결과가 없으면 기본 설정으로 재시도
            if not best_text.strip():
                try:
                    best_text = pytesseract.image_to_string(processed_image, config=r'--oem 3 --psm 3 -l kor+eng')
                except Exception as e:
                    st.warning(f"페이지 {i+1} OCR 실패: {str(e)}")
                    continue
            
            # 텍스트 후처리
            if best_text.strip():
                processed_text = postprocess_text(best_text)
                extracted_text += f"\n--- 페이지 {i+1} ---\n{processed_text}\n"
        
        if extracted_text.strip():
            st.success(f"🎉 고급 OCR 완료: {len(extracted_text)} 글자 추출")
        else:
            st.warning("⚠️ OCR로 텍스트를 추출할 수 없었습니다.")
        
        return extracted_text
        
    except Exception as e:
        return f"고급 OCR 처리 중 오류: {str(e)}"

# OCR 텍스트 추출 함수 (기존 함수를 고급 버전으로 교체)
def extract_text_with_ocr(pdf_bytes):
    """OCR을 사용하여 이미지 기반 PDF에서 텍스트 추출 (고급 버전)"""
    return extract_text_with_advanced_ocr(pdf_bytes)

# 로컬 PDF 처리 함수 (기본 텍스트 추출 로직 개선)
def process_pdf_locally(request_data, progress_callback=None):
    """PDF를 로컬에서 직접 처리하는 함수 (기본 텍스트 추출 로직 개선)"""
    try:
        import pdfplumber
        import openai
        
        if progress_callback:
            progress_callback("PDF 파일 디코딩 중...", 0.1)
        
        # base64 디코딩
        file_content = base64.b64decode(request_data['file_content'])
        
        if progress_callback:
            progress_callback("기본 텍스트 추출 중...", 0.2)
        
        # PDF 텍스트 추출 (기본 방식) - 항상 실행
        extracted_text = ""
        basic_extraction_success = False
        
        try:
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                total_pages = len(pdf.pages)
                st.info(f"📄 PDF 페이지 수: {total_pages}")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            extracted_text += f"\n--- 페이지 {i+1} ---\n{text}\n"
                            basic_extraction_success = True
                    except Exception as page_error:
                        st.warning(f"페이지 {i+1} 텍스트 추출 실패: {str(page_error)}")
                        continue
                
                if basic_extraction_success:
                    st.success(f"✅ 기본 텍스트 추출 성공: {len(extracted_text)} 글자")
                else:
                    st.warning("⚠️ 기본 텍스트 추출에서 텍스트를 찾을 수 없습니다.")
                    
        except Exception as e:
            st.error(f"❌ 기본 텍스트 추출 실패: {str(e)}")
            basic_extraction_success = False
        
        # 사용자가 OCR 옵션을 선택한 경우에만 OCR 실행
        if request_data['options'].get('use_ocr') and OCR_AVAILABLE:
            if progress_callback:
                progress_callback("고급 OCR을 사용한 텍스트 추출 중...", 0.4)
            
            st.info("🔍 고급 OCR 텍스트 추출을 시작합니다...")
            ocr_text = extract_text_with_advanced_ocr(file_content)
            
            if ocr_text and not ocr_text.startswith("고급 OCR 처리 중 오류"):
                # OCR 결과가 있으면 기본 텍스트와 결합하거나 대체
                if len(extracted_text.strip()) < 100:
                    # 기본 텍스트가 부족하면 OCR 텍스트로 대체
                    extracted_text = ocr_text
                    st.info("📷 고급 OCR을 사용하여 텍스트를 추출했습니다.")
                else:
                    # 기본 텍스트가 충분하면 OCR 텍스트를 추가
                    extracted_text += "\n\n=== 고급 OCR 추출 텍스트 ===\n" + ocr_text
                    st.info("📷 기본 텍스트 추출과 고급 OCR을 모두 사용했습니다.")
            else:
                st.warning("고급 OCR 텍스트 추출에 실패했습니다.")
        
        # 최종 텍스트 길이 확인
        final_text_length = len(extracted_text.strip())
        if final_text_length == 0:
            st.error("❌ 텍스트 추출에 실패했습니다. PDF가 이미지 기반일 수 있습니다. OCR 옵션을 사용해보세요.")
            extracted_text = "텍스트 추출에 실패했습니다. 이 PDF는 이미지 기반일 수 있습니다."
        else:
            st.success(f"✅ 최종 추출된 텍스트: {final_text_length:,} 글자")
        
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

# 모바일 반응형 CSS 추가 (배경색 문제 수정)
st.markdown("""
<style>
    /* 기본 스타일 리셋 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* 정보 카드 스타일 개선 (가독성 문제 해결) */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    .info-card strong {
        color: #ffffff;
        font-weight: 600;
    }
    
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
        
        .info-card {
            padding: 1rem;
            margin: 0.5rem 0;
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
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    }
    
    /* 성공 메시지 스타일 */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    
    /* 경고 메시지 스타일 */
    .stWarning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
    
    /* 오류 메시지 스타일 */
    .stError {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    
    /* 사이드바 스타일 개선 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* 모바일에서 사이드바 자동 축소 */
    @media (max-width: 768px) {
        .css-1d391kg {
            width: 0px;
        }
    }
    
    /* 버튼 스타일 개선 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* 파일 업로더 스타일 */
    .stFileUploader > div > div {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9ff;
    }
    
    /* 텍스트 영역 스타일 */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.title("📄 HangulPDF AI Converter")
st.markdown("한글 PDF 문서를 AI가 활용하기 쉬운 형태로 변환하고 분석하는 도구입니다.")

# OCR 상태 표시
if OCR_AVAILABLE:
    st.success("🔍 고급 OCR 기능이 활성화되었습니다. 이미지 기반 PDF도 고정밀도로 처리할 수 있습니다.")
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
- 🔍 **고급 OCR**: 이미지 기반 PDF 고정밀도 처리
- 🤖 **AI 요약**: OpenAI를 활용한 문서 요약
- ❓ **Q&A 생성**: 자동 질문-답변 생성
- 📱 **모바일 지원**: 다양한 디바이스에서 사용 가능
""")

# 사이드바 - OCR 개선 정보
if OCR_AVAILABLE:
    st.sidebar.header("🔬 고급 OCR 기능")
    st.sidebar.markdown("""
    **정확도 향상 기술:**
    - 🎯 **고해상도 처리** (400 DPI)
    - 🖼️ **고급 이미지 전처리**
    - 🔄 **다단계 OCR 처리**
    - 📝 **텍스트 후처리**
    - 📐 **기울기 자동 보정**
    - 🎨 **대비 및 선명도 최적화**
    """)

# 메인 컨텐츠
tab1, tab2, tab3 = st.tabs(["📄 PDF 업로드 & 변환", "📊 분석 결과", "🔗 공유 & 내보내기"])

with tab1:
    st.header("PDF 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=['pdf'],
        help="한글로 작성된 PDF 문서를 업로드하세요. 고급 OCR로 이미지 기반 PDF도 고정밀도 처리합니다."
    )
    
    if uploaded_file is not None:
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
        
        # 파일 정보 표시 (가독성 개선)
        col1, col2 = st.columns([3, 1])
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
        
        # 반응형 레이아웃
        col1, col2 = st.columns(2)
        with col1:
            extract_text = st.checkbox("📝 텍스트 추출", value=True)
            generate_summary = st.checkbox("📋 요약 생성", value=True, disabled=not openai_api_key)
            # OCR 옵션 추가
            use_ocr = st.checkbox(
                "🔍 고급 OCR 사용", 
                value=False, 
                disabled=not OCR_AVAILABLE,
                help="고정밀도 이미지 전처리와 다단계 OCR로 이미지 기반 PDF에서 정확한 텍스트를 추출합니다."
            )
        with col2:
            generate_qa = st.checkbox("❓ 질문-답변 생성", value=False, disabled=not openai_api_key)
            clean_text = st.checkbox("🧹 텍스트 정제", value=True)
        
        # OCR 옵션 설명
        if OCR_AVAILABLE:
            if use_ocr:
                st.info("🔍 **고급 OCR 모드**: 고해상도 이미지 전처리, 다단계 OCR 처리, 텍스트 후처리로 최고 정확도를 제공합니다. 처리 시간이 더 오래 걸릴 수 있습니다.")
            else:
                st.info("📄 **기본 모드**: 텍스트 기반 PDF에서만 텍스트를 추출합니다. 빠른 처리가 가능합니다.")
        
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
                            "clean_text": clean_text,
                            "use_ocr": use_ocr  # OCR 옵션 추가
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
            # 반응형 레이아웃
            col1, col2 = st.columns(2)
            
            with col1:
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
            
            with col2:
                st.subheader("🤖 AI 모델 연동")
            
            if 'extracted_text' in result:
                # ChatGPT 프롬프트
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
st.markdown("🔬 **고급 OCR**: 고해상도 이미지 전처리 + 다단계 OCR 처리 + 텍스트 후처리로 최고 정확도 제공")

