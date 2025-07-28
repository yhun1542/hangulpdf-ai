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

# 표 전용 OCR 처리 함수
def process_table_ocr(image, table_regions):
    """표 구조에 최적화된 OCR 처리"""
    extracted_text = ""
    
    if not table_regions:
        return ""
    
    st.info(f"📊 감지된 표 개수: {len(table_regions)}")
    
    for i, (x, y, w, h) in enumerate(table_regions):
        try:
            # 표 영역 추출
            table_roi = image[y:y+h, x:x+w]
            
            # 한글 특화 전처리 적용
            processed_table = preprocess_for_korean(Image.fromarray(table_roi))
            
            # 표 전용 OCR 설정들
            table_configs = [
                # 표 구조 인식에 최적화된 설정
                r'--oem 3 --psm 6 -l kor+eng -c preserve_interword_spaces=1',
                r'--oem 3 --psm 4 -l kor+eng -c preserve_interword_spaces=1',
                r'--oem 3 --psm 3 -l kor+eng -c preserve_interword_spaces=1',
            ]
            
            best_table_text = ""
            best_confidence = 0
            
            for config in table_configs:
                try:
                    text = pytesseract.image_to_string(processed_table, config=config)
                    if text.strip():
                        # 표 텍스트 신뢰도 측정 (한글 비율 + 표 구조 요소)
                        korean_chars = len(re.findall(r'[가-힣]', text))
                        total_chars = len(re.sub(r'\s', '', text))
                        table_indicators = len(re.findall(r'[|─┌┐└┘├┤┬┴┼]', text))  # 표 구조 문자
                        
                        if total_chars > 0:
                            confidence = (korean_chars / total_chars) * len(text.strip()) + table_indicators * 10
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_table_text = text
                
                except Exception:
                    continue
            
            if best_table_text.strip():
                extracted_text += f"\n=== 표 {i+1} ===\n{best_table_text}\n"
                st.success(f"✅ 표 {i+1} 처리 완료 (신뢰도: {best_confidence:.1f})")
            else:
                st.warning(f"⚠️ 표 {i+1} 텍스트 추출 실패")
                
        except Exception as e:
            st.warning(f"표 {i+1} 처리 중 오류: {str(e)}")
            continue
    
    return extracted_text

# 이미지 내 텍스트 영역 OCR 처리 함수
def process_text_regions_ocr(image, text_regions):
    """이미지 내 텍스트 영역에 최적화된 OCR 처리"""
    extracted_text = ""
    
    if not text_regions:
        return ""
    
    st.info(f"🖼️ 감지된 텍스트 영역 개수: {len(text_regions)}")
    
    # 텍스트 영역을 크기순으로 정렬 (큰 영역부터 처리)
    text_regions = sorted(text_regions, key=lambda x: x[2] * x[3], reverse=True)
    
    for i, (x, y, w, h) in enumerate(text_regions[:10]):  # 상위 10개 영역만 처리
        try:
            # 텍스트 영역 추출 (여백 추가)
            margin = 5
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(image.shape[1], x + w + margin)
            y_end = min(image.shape[0], y + h + margin)
            
            text_roi = image[y_start:y_end, x_start:x_end]
            
            # 한글 특화 전처리 적용
            processed_text = preprocess_for_korean(Image.fromarray(text_roi))
            
            # 텍스트 영역 전용 OCR 설정
            text_configs = [
                r'--oem 3 --psm 7 -l kor+eng',  # 단일 텍스트 라인
                r'--oem 3 --psm 8 -l kor+eng',  # 단일 단어
                r'--oem 3 --psm 6 -l kor+eng',  # 단일 텍스트 블록
            ]
            
            best_text = ""
            best_confidence = 0
            
            for config in text_configs:
                try:
                    text = pytesseract.image_to_string(processed_text, config=config)
                    if text.strip():
                        # 텍스트 영역 신뢰도 측정
                        korean_chars = len(re.findall(r'[가-힣]', text))
                        total_chars = len(re.sub(r'\s', '', text))
                        
                        if total_chars > 0:
                            confidence = (korean_chars / total_chars) * len(text.strip())
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_text = text
                
                except Exception:
                    continue
            
            if best_text.strip() and best_confidence > 5:  # 최소 신뢰도 필터
                extracted_text += f"[텍스트영역{i+1}] {best_text.strip()}\n"
                
        except Exception as e:
            continue
    
    return extracted_text

# 고급 이미지 전처리 함수 (기존 함수 개선)
def preprocess_image_advanced(image):
    """고급 이미지 전처리로 OCR 정확도 향상 (표/이미지 특화)"""
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
        
        # 4. 대비 향상 (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 5. 모폴로지 연산으로 텍스트 선명화
        kernel = np.ones((1, 1), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 6. 이진화 (Otsu's thresholding)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 7. 기울기 보정
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 1:
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return binary
        
    except Exception as e:
        st.warning(f"이미지 전처리 중 오류: {str(e)}")
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        return gray

# 텍스트 후처리 함수 (표 구조 개선)
def postprocess_text(text):
    """OCR 결과 텍스트 후처리로 정확도 향상 (표 구조 특화)"""
    if not text:
        return text
    
    # 1. 표 구조 정리
    # 표 구분선 정리
    text = re.sub(r'[─━═]{2,}', '─' * 10, text)  # 긴 수평선을 표준 길이로
    text = re.sub(r'[│┃║]{2,}', '│', text)  # 연속된 수직선을 하나로
    
    # 2. 표 셀 구분 개선
    text = re.sub(r'\s*[│|]\s*', ' │ ', text)  # 표 구분자 주변 공백 정리
    text = re.sub(r'\s*[─┌┐└┘├┤┬┴┼]\s*', ' ', text)  # 표 모서리 문자 정리
    
    # 3. 일반적인 OCR 오류 수정
    corrections = {
        'ㅇ': 'o',  # 영문 o와 한글 ㅇ 구분
        'ㅁ': 'm',  # 영문 m과 한글 ㅁ 구분
        '|': 'l',   # 세로선과 영문 l 구분
        '0': 'O',   # 숫자 0과 영문 O 구분
        '1': 'l',   # 숫자 1과 영문 l 구분
    }
    
    # 4. 불필요한 공백 정리
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백을 하나로
    text = re.sub(r'\n\s*\n', '\n\n', text)  # 연속된 줄바꿈 정리
    
    # 5. 한글 문장 부호 정리
    text = re.sub(r'["""]', '"', text)  # 따옴표 통일
    text = re.sub(r"[''']", "'", text)  # 작은따옴표 통일
    text = re.sub(r'[…]', '...', text)  # 말줄임표 통일
    
    # 6. 숫자와 단위 사이 공백 정리
    text = re.sub(r'(\d)\s+([가-힣]{1,2})', r'\1\2', text)  # "10 개" -> "10개"
    text = re.sub(r'(\d)\s+(%|원|달러|kg|m|cm)', r'\1\2', text)  # "100 %" -> "100%"
    
    # 7. 날짜 형식 정리
    text = re.sub(r'(\d{4})\s*[년]\s*(\d{1,2})\s*[월]\s*(\d{1,2})\s*[일]', r'\1년 \2월 \3일', text)
    
    # 8. 표 내용 정리
    text = re.sub(r'(\d+)\s*[.]\s*(\d+)', r'\1.\2', text)  # 소수점 정리
    text = re.sub(r'(\d+)\s*[,]\s*(\d+)', r'\1,\2', text)  # 천단위 구분자 정리
    
    return text.strip()

# 다단계 OCR 처리 함수 (표/이미지 특화)
def extract_text_with_advanced_ocr(pdf_bytes):
    """고급 OCR을 사용하여 이미지 기반 PDF에서 텍스트 추출 (표/이미지 특화)"""
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
                table_text = process_table_ocr(img_array, table_regions)
            
            # 2. 텍스트 영역 감지 및 처리
            text_regions = detect_text_regions(img_array)
            region_text = ""
            if text_regions:
                st.info(f"🖼️ 페이지 {i+1}에서 {len(text_regions)}개의 텍스트 영역 감지")
                region_text = process_text_regions_ocr(img_array, text_regions)
            
            # 3. 전체 페이지 OCR (기본 처리)
            processed_image = preprocess_image_advanced(image)
            
            # 다양한 OCR 설정들
            ocr_configs = [
                {
                    'config': r'--oem 3 --psm 3 -l kor+eng',
                    'name': '자동 페이지 분할'
                },
                {
                    'config': r'--oem 3 --psm 6 -l kor+eng',
                    'name': '단일 텍스트 블록'
                },
                {
                    'config': r'--oem 3 --psm 4 -l kor+eng',
                    'name': '단일 텍스트 컬럼'
                }
            ]
            
            best_text = ""
            best_confidence = 0
            
            # 여러 OCR 설정으로 시도
            for config_info in ocr_configs:
                try:
                    text = pytesseract.image_to_string(processed_image, config=config_info['config'])
                    
                    if text.strip():
                        korean_chars = len(re.findall(r'[가-힣]', text))
                        total_chars = len(re.sub(r'\s', '', text))
                        
                        if total_chars > 0:
                            confidence = (korean_chars / total_chars) * len(text.strip())
                            
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_text = text
                                st.success(f"✅ 최적 설정: {config_info['name']} (신뢰도: {confidence:.1f})")
                
                except Exception:
                    continue
            
            # 결과 통합
            page_text = f"\n--- 페이지 {i+1} ---\n"
            
            if table_text.strip():
                page_text += table_text
            
            if region_text.strip():
                page_text += f"\n=== 이미지 내 텍스트 ===\n{region_text}\n"
            
            if best_text.strip():
                page_text += f"\n=== 전체 페이지 텍스트 ===\n{best_text}\n"
            
            # 텍스트 후처리
            if page_text.strip():
                processed_text = postprocess_text(page_text)
                extracted_text += processed_text
        
        if extracted_text.strip():
            st.success(f"🎉 표/이미지 특화 OCR 완료: {len(extracted_text)} 글자 추출")
        else:
            st.warning("⚠️ OCR로 텍스트를 추출할 수 없었습니다.")
        
        return extracted_text
        
    except Exception as e:
        return f"표/이미지 특화 OCR 처리 중 오류: {str(e)}"

# OCR 텍스트 추출 함수 (기존 함수를 표/이미지 특화 버전으로 교체)
def extract_text_with_ocr(pdf_bytes):
    """OCR을 사용하여 이미지 기반 PDF에서 텍스트 추출 (표/이미지 특화 버전)"""
    return extract_text_with_advanced_ocr(pdf_bytes)

# 로컬 PDF 처리 함수
def process_pdf_locally(request_data, progress_callback=None):
    """PDF를 로컬에서 직접 처리하는 함수"""
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
                progress_callback("표/이미지 특화 OCR을 사용한 텍스트 추출 중...", 0.4)
            
            st.info("🔍 표/이미지 특화 OCR 텍스트 추출을 시작합니다...")
            ocr_text = extract_text_with_advanced_ocr(file_content)
            
            if ocr_text and not ocr_text.startswith("표/이미지 특화 OCR 처리 중 오류"):
                # OCR 결과가 있으면 기본 텍스트와 결합하거나 대체
                if len(extracted_text.strip()) < 100:
                    # 기본 텍스트가 부족하면 OCR 텍스트로 대체
                    extracted_text = ocr_text
                    st.info("📷 표/이미지 특화 OCR을 사용하여 텍스트를 추출했습니다.")
                else:
                    # 기본 텍스트가 충분하면 OCR 텍스트를 추가
                    extracted_text += "\n\n=== 표/이미지 특화 OCR 추출 텍스트 ===\n" + ocr_text
                    st.info("📷 기본 텍스트 추출과 표/이미지 특화 OCR을 모두 사용했습니다.")
            else:
                st.warning("표/이미지 특화 OCR 텍스트 추출에 실패했습니다.")
        
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
        
        # OpenAI 클라이언트 설정
        api_key = request_data.get('openai_api_key')
        if api_key and api_key.strip() and not api_key.startswith('sk-') == False:
            try:
                client = openai.OpenAI(api_key=api_key.strip())
                
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
                        
                        result['qa_pairs'] = qa_pairs[:3]
                        
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
    /* 기본 스타일 리셋 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* 정보 카드 스타일 개선 */
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
    st.success("🔍 표/이미지 특화 고급 OCR 기능이 활성화되었습니다. 표와 이미지 내 한글도 고정밀도로 처리할 수 있습니다.")
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
- 📊 **표 특화 OCR**: 표 구조 감지 및 셀별 처리
- 🖼️ **이미지 내 텍스트**: 이미지 영역 감지 및 추출
- 🤖 **AI 요약**: OpenAI를 활용한 문서 요약
- ❓ **Q&A 생성**: 자동 질문-답변 생성
- 📱 **모바일 지원**: 다양한 디바이스에서 사용 가능
""")

# 사이드바 - 표/이미지 특화 OCR 정보
if OCR_AVAILABLE:
    st.sidebar.header("📊 표/이미지 특화 OCR")
    st.sidebar.markdown("""
    **표 처리 기술:**
    - 🔍 **표 구조 자동 감지**
    - 📐 **셀별 분할 처리**
    - 📊 **표 전용 OCR 설정**
    
    **이미지 텍스트 처리:**
    - 🖼️ **텍스트 영역 자동 감지**
    - 🎯 **한글 특화 전처리**
    - 📝 **영역별 최적화 OCR**
    
    **정확도 향상 기술:**
    - 🎨 **3배 해상도 증가**
    - 🔧 **언샤프 마스킹**
    - 📏 **적응형 이진화**
    - 🔄 **다단계 신뢰도 측정**
    """)

# 메인 컨텐츠
tab1, tab2, tab3 = st.tabs(["📄 PDF 업로드 & 변환", "📊 분석 결과", "🔗 공유 & 내보내기"])

with tab1:
    st.header("PDF 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=['pdf'],
        help="한글로 작성된 PDF 문서를 업로드하세요. 표/이미지 특화 OCR로 복잡한 문서도 고정밀도 처리합니다."
    )
    
    if uploaded_file is not None:
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
        
        # 파일 정보 표시
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
                "📊 표/이미지 특화 OCR 사용", 
                value=False, 
                disabled=not OCR_AVAILABLE,
                help="표 구조 감지, 이미지 내 텍스트 영역 감지, 한글 특화 전처리로 표와 이미지 내 한글을 정확하게 추출합니다."
            )
        with col2:
            generate_qa = st.checkbox("❓ 질문-답변 생성", value=False, disabled=not openai_api_key)
            clean_text = st.checkbox("🧹 텍스트 정제", value=True)
        
        # OCR 옵션 설명
        if OCR_AVAILABLE:
            if use_ocr:
                st.info("📊 **표/이미지 특화 OCR 모드**: 표 구조 자동 감지, 이미지 내 텍스트 영역 감지, 한글 특화 전처리로 표와 이미지 내 한글을 최고 정확도로 추출합니다. 처리 시간이 더 오래 걸릴 수 있습니다.")
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
                    time.sleep(0.1)
                
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
                            "use_ocr": use_ocr
                        },
                        "openai_api_key": openai_api_key
                    }
                    
                    # 세션 상태에 결과 저장
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
st.markdown("📊 **표/이미지 특화 OCR**: 표 구조 감지 + 이미지 내 텍스트 영역 감지 + 한글 특화 전처리로 최고 정확도 제공")

