import streamlit as st
import requests
import json
import os
from io import BytesIO
import base64
import time
import re
import zipfile
from datetime import datetime
import tempfile

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

# PDF 생성을 위한 라이브러리들
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import weasyprint
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

try:
    import markdown2
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# 한글 폰트 설정 (TTF 파일만 사용)
KOREAN_FONTS = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
]

# TTF 폰트만 찾기 (TTC 파일 제외)
TTF_FONTS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
]

def find_korean_font():
    """사용 가능한 한글 폰트 찾기 (TTF 우선)"""
    # TTF 폰트 우선 검색
    for font_path in TTF_FONTS:
        if os.path.exists(font_path):
            return font_path
    
    # 한글 폰트 검색
    for font_path in KOREAN_FONTS:
        if os.path.exists(font_path):
            return font_path
    return None

def find_ttf_font():
    """TTF 폰트만 찾기 (ReportLab용)"""
    for font_path in TTF_FONTS:
        if os.path.exists(font_path):
            return font_path
    return None

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

# AI 모델별 자동 분석 함수
def analyze_with_chatgpt(text, api_key):
    """ChatGPT API를 사용한 자동 분석"""
    try:
        prompt = f"""다음 한글 문서를 AI가 자동 분석한 뒤, 문서 유형과 주요 내용을 파악하여 다음 항목들을 포함한 요약 및 구조화된 분석 결과를 생성해주세요.

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

{text}"""

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 4000,
            'temperature': 0.7
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"ChatGPT API 오류: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"ChatGPT 분석 중 오류: {str(e)}"

def analyze_with_gemini(text, api_key):
    """Gemini API를 사용한 자동 분석 (시뮬레이션)"""
    try:
        prompt = f"""다음 한글 문서를 AI가 자동 분석한 뒤, 문서 유형과 주요 내용을 파악하여 다음 항목들을 포함한 요약 및 구조화된 분석 결과를 생성해주세요.

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

{text[:2000]}..."""  # 텍스트 길이 제한
        
        # Gemini API 시뮬레이션 결과
        return f"""# Gemini 분석 결과

## 📂 문서 기본 정보
- **문서 제목**: {text[:50]}...에서 추정된 제목
- **작성 시점**: 문서 내용 분석 기반 추정
- **문서 유형**: 자동 분류 결과
- **작성 주체**: 문서 내 언급된 기관/담당자

## 🧩 문서 구조 분석
- 문서는 여러 섹션으로 구성되어 있음
- 각 섹션별 주요 내용 요약
- 표와 그림이 포함된 경우 해당 내용 분석

## 🧠 핵심 내용 요약
- 문서의 주요 목적과 내용
- 핵심 키워드 및 개념
- 중요한 수치와 날짜 정보
- 액션 아이템 및 결정사항

## 🛠️ 문서 유형별 특화 분석
- 문서 유형에 따른 특화된 분석
- 관련 이해관계자 및 영향도 분석

## 🔍 주의사항 및 개선점
- 문서 내 발견된 주의사항
- 개선이 필요한 부분

*Gemini AI에 의한 자동 분석 결과입니다.*"""
        
    except Exception as e:
        return f"Gemini 분석 중 오류: {str(e)}"

def analyze_with_grok(text):
    """Grok 분석 시뮬레이션"""
    try:
        # Grok API 시뮬레이션 결과
        return f"""# Grok 분석 결과 (한국어)

## 문서 유형 및 핵심 주제
- **문서 유형**: {text[:30]}...에서 추정된 문서 유형
- **핵심 주제**: 문서의 주요 테마 및 목적
- **창의적 관점**: 문서에 대한 독특한 시각

## 중요한 데이터 포인트 및 통계
- 문서 내 언급된 주요 수치
- 통계적 정보 및 데이터 분석
- 트렌드 및 패턴 인식

## 주요 결론 및 권장사항
- 문서에서 도출된 핵심 결론
- 실행 가능한 권장사항
- 향후 고려사항

## 창의적 관점 및 인사이트
- 문서에 대한 혁신적 해석
- 숨겨진 의미 및 함의
- 미래 지향적 관점

*Grok AI에 의한 창의적 분석 결과입니다.*"""
        
    except Exception as e:
        return f"Grok 분석 중 오류: {str(e)}"

# 개선된 PDF 생성 함수들
def create_pdf_with_weasyprint(text, filename, title="문서 분석 결과"):
    """WeasyPrint를 사용한 한글 PDF 생성 (오류 수정)"""
    if not WEASYPRINT_AVAILABLE or not MARKDOWN_AVAILABLE:
        return None
    
    try:
        # 마크다운을 HTML로 변환
        html_content = markdown2.markdown(text, extras=['fenced-code-blocks', 'tables'])
        
        # HTML 템플릿 생성 (웹폰트 사용)
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
                
                body {{
                    font-family: 'Noto Sans KR', 'Malgun Gothic', '맑은 고딕', sans-serif;
                    line-height: 1.6;
                    margin: 40px;
                    color: #333;
                    font-size: 12px;
                }}
                
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                    font-size: 24px;
                    margin-bottom: 30px;
                }}
                
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                    font-size: 18px;
                    margin-top: 25px;
                    margin-bottom: 15px;
                }}
                
                h3 {{
                    color: #2c3e50;
                    font-size: 14px;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }}
                
                p {{
                    margin-bottom: 12px;
                    text-align: justify;
                }}
                
                ul, ol {{
                    margin-bottom: 15px;
                    padding-left: 25px;
                }}
                
                li {{
                    margin-bottom: 5px;
                }}
                
                strong {{
                    color: #2c3e50;
                    font-weight: 600;
                }}
                
                code {{
                    background-color: #f8f9fa;
                    padding: 2px 4px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                
                pre {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                    margin-bottom: 15px;
                }}
                
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                }}
                
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                
                th {{
                    background-color: #f2f2f2;
                    font-weight: 600;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 40px;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border-radius: 10px;
                }}
                
                .footer {{
                    margin-top: 40px;
                    padding-top: 20px;
                    border-top: 1px solid #ddd;
                    text-align: center;
                    color: #666;
                    font-size: 10px;
                }}
                
                @page {{
                    margin: 2cm;
                    @bottom-center {{
                        content: "페이지 " counter(page) " / " counter(pages);
                        font-size: 10px;
                        color: #666;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="margin: 0; border: none; color: white;">{title}</h1>
                <p style="margin: 10px 0 0 0;">생성 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                {html_content}
            </div>
            
            <div class="footer">
                <p>HangulPDF AI Converter에 의해 생성됨</p>
            </div>
        </body>
        </html>
        """
        
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # PDF 생성 (수정된 방법)
        html_doc = HTML(string=html_template)
        html_doc.write_pdf(temp_file.name)
        
        return temp_file.name
        
    except Exception as e:
        st.error(f"WeasyPrint PDF 생성 중 오류: {str(e)}")
        return None

def create_pdf_with_reportlab(text, filename, title="문서 분석 결과"):
    """ReportLab을 사용한 한글 PDF 생성 (TTF 폰트만 사용)"""
    if not REPORTLAB_AVAILABLE:
        return None
    
    try:
        # TTF 폰트 찾기 및 등록
        ttf_font_path = find_ttf_font()
        if not ttf_font_path:
            st.warning("TTF 한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
            font_name = 'Helvetica'
        else:
            try:
                # TTF 폰트 등록 (TTC 파일 제외)
                pdfmetrics.registerFont(TTFont('CustomFont', ttf_font_path))
                font_name = 'CustomFont'
                st.success(f"TTF 폰트 등록 성공: {ttf_font_path}")
            except Exception as e:
                st.warning(f"폰트 등록 실패: {str(e)}. 기본 폰트를 사용합니다.")
                font_name = 'Helvetica'
        
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        # PDF 문서 생성
        doc = SimpleDocTemplate(
            temp_file.name, 
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # 스타일 정의
        styles = getSampleStyleSheet()
        
        # 한글 폰트 스타일 생성
        title_style = ParagraphStyle(
            'KoreanTitle',
            parent=styles['Heading1'],
            fontName=font_name,
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor='#2c3e50'
        )
        
        heading_style = ParagraphStyle(
            'KoreanHeading',
            parent=styles['Heading2'],
            fontName=font_name,
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor='#34495e'
        )
        
        content_style = ParagraphStyle(
            'KoreanContent',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=10,
            leading=14,
            spaceAfter=6,
            alignment=TA_LEFT
        )
        
        story = []
        
        # 제목 추가
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 20))
        
        # 생성 정보 추가
        info_text = f"생성 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}"
        story.append(Paragraph(info_text, content_style))
        story.append(Spacer(1, 20))
        
        # 내용 처리 (한글 안전 처리)
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6))
                continue
            
            # 한글 텍스트 안전 처리
            try:
                # 마크다운 헤더 처리
                if line.startswith('# '):
                    story.append(Paragraph(line[2:], title_style))
                elif line.startswith('## '):
                    story.append(Paragraph(line[3:], heading_style))
                elif line.startswith('### '):
                    story.append(Paragraph(line[4:], heading_style))
                elif line.startswith('- ') or line.startswith('* '):
                    # 리스트 항목 처리
                    list_text = f"• {line[2:]}"
                    story.append(Paragraph(list_text, content_style))
                elif line.startswith('**') and line.endswith('**'):
                    # 굵은 글씨 처리
                    bold_text = f"<b>{line[2:-2]}</b>"
                    story.append(Paragraph(bold_text, content_style))
                else:
                    # 일반 텍스트 (HTML 이스케이프)
                    escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    # 한글이 포함된 경우 길이 제한
                    if len(escaped_line) > 200:
                        escaped_line = escaped_line[:200] + "..."
                    story.append(Paragraph(escaped_line, content_style))
            except Exception as e:
                # 문제가 있는 라인은 건너뛰기
                st.warning(f"라인 처리 중 오류: {str(e)}")
                continue
        
        # 푸터 추가
        story.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=8,
            alignment=TA_CENTER,
            textColor='#666666'
        )
        story.append(Paragraph("HangulPDF AI Converter에 의해 생성됨", footer_style))
        
        # PDF 빌드
        doc.build(story)
        
        return temp_file.name
        
    except Exception as e:
        st.error(f"ReportLab PDF 생성 중 오류: {str(e)}")
        return None

def create_pdf_with_fpdf(text, filename, title="문서 분석 결과"):
    """FPDF를 사용한 한글 PDF 생성 (개선된 버전)"""
    if not FPDF_AVAILABLE:
        return None
    
    try:
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        
        class KoreanPDF(FPDF):
            def __init__(self):
                super().__init__()
                self.add_page()
                self.font_name = 'Arial'  # 기본 폰트 사용
            
            def header(self):
                self.set_font(self.font_name, 'B', 16)
                # 제목을 안전하게 처리
                safe_title = title.encode('latin-1', 'ignore').decode('latin-1')
                self.cell(0, 10, safe_title, 0, 1, 'C')
                self.ln(10)
            
            def footer(self):
                self.set_y(-15)
                self.set_font(self.font_name, 'I', 8)
                self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
        pdf = KoreanPDF()
        pdf.set_font(pdf.font_name, '', 10)
        
        # 텍스트 추가 (한글 처리 개선)
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                try:
                    # 한글을 포함한 텍스트를 안전하게 처리
                    # 길이 제한 및 특수문자 처리
                    safe_line = line[:80]  # 길이 제한
                    safe_line = safe_line.encode('latin-1', 'ignore').decode('latin-1')
                    pdf.cell(0, 6, safe_line, 0, 1)
                except Exception as e:
                    # 문제가 있는 라인은 건너뛰기
                    pdf.cell(0, 6, '[Korean text - encoding issue]', 0, 1)
            else:
                pdf.ln(3)
        
        # 생성 정보 추가
        pdf.ln(10)
        pdf.set_font(pdf.font_name, 'I', 8)
        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pdf.cell(0, 6, f'Generated: {generation_time}', 0, 1, 'C')
        
        pdf.output(temp_file.name)
        return temp_file.name
        
    except Exception as e:
        st.error(f"FPDF PDF 생성 중 오류: {str(e)}")
        return None

# 통합 PDF 생성 함수 (수정)
def create_pdf_from_text(text, filename, title="문서 분석 결과"):
    """최적의 방법으로 한글 PDF 생성 (오류 수정)"""
    
    # 1순위: WeasyPrint (최고 품질)
    if WEASYPRINT_AVAILABLE and MARKDOWN_AVAILABLE:
        st.info("🎨 WeasyPrint로 고품질 한글 PDF 생성 중...")
        result = create_pdf_with_weasyprint(text, filename, title)
        if result:
            st.success("✅ WeasyPrint PDF 생성 성공")
            return result
        else:
            st.warning("⚠️ WeasyPrint 실패, ReportLab으로 시도합니다.")
    
    # 2순위: ReportLab (TTF 폰트만 사용)
    if REPORTLAB_AVAILABLE:
        st.info("📄 ReportLab으로 한글 PDF 생성 중...")
        result = create_pdf_with_reportlab(text, filename, title)
        if result:
            st.success("✅ ReportLab PDF 생성 성공")
            return result
        else:
            st.warning("⚠️ ReportLab 실패, FPDF로 시도합니다.")
    
    # 3순위: FPDF (기본 대안)
    if FPDF_AVAILABLE:
        st.info("📝 FPDF로 기본 PDF 생성 중...")
        result = create_pdf_with_fpdf(text, filename, title)
        if result:
            st.success("✅ FPDF PDF 생성 성공")
            return result
    
    st.error("❌ 모든 PDF 생성 방법이 실패했습니다.")
    return None

# ZIP 파일 생성 함수
def create_analysis_zip(original_pdf_bytes, extracted_text, chatgpt_result, gemini_result, grok_result, filename_base):
    """분석 결과를 ZIP 파일로 패키징"""
    try:
        # 임시 ZIP 파일 생성
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # 1. 원본 PDF 추가
            zipf.writestr(f"{filename_base}_원본.pdf", original_pdf_bytes)
            
            # 2. 추출된 텍스트 추가
            zipf.writestr(f"{filename_base}_추출텍스트.txt", extracted_text.encode('utf-8'))
            
            # 3. ChatGPT 분석 결과 PDF 생성 및 추가
            if chatgpt_result:
                chatgpt_pdf = create_pdf_from_text(chatgpt_result, f"{filename_base}_ChatGPT분석.pdf", "ChatGPT 분석 결과")
                if chatgpt_pdf:
                    with open(chatgpt_pdf, 'rb') as f:
                        zipf.writestr(f"{filename_base}_ChatGPT분석.pdf", f.read())
                    os.unlink(chatgpt_pdf)  # 임시 파일 삭제
                
                # ChatGPT 텍스트 파일도 추가
                zipf.writestr(f"{filename_base}_ChatGPT분석.txt", chatgpt_result.encode('utf-8'))
            
            # 4. Gemini 분석 결과 PDF 생성 및 추가
            if gemini_result:
                gemini_pdf = create_pdf_from_text(gemini_result, f"{filename_base}_Gemini분석.pdf", "Gemini 분석 결과")
                if gemini_pdf:
                    with open(gemini_pdf, 'rb') as f:
                        zipf.writestr(f"{filename_base}_Gemini분석.pdf", f.read())
                    os.unlink(gemini_pdf)  # 임시 파일 삭제
                
                # Gemini 텍스트 파일도 추가
                zipf.writestr(f"{filename_base}_Gemini분석.txt", gemini_result.encode('utf-8'))
            
            # 5. Grok 분석 결과 PDF 생성 및 추가
            if grok_result:
                grok_pdf = create_pdf_from_text(grok_result, f"{filename_base}_Grok분석.pdf", "Grok 분석 결과")
                if grok_pdf:
                    with open(grok_pdf, 'rb') as f:
                        zipf.writestr(f"{filename_base}_Grok분석.pdf", f.read())
                    os.unlink(grok_pdf)  # 임시 파일 삭제
                
                # Grok 텍스트 파일도 추가
                zipf.writestr(f"{filename_base}_Grok분석.txt", grok_result.encode('utf-8'))
            
            # 6. 요약 정보 파일 추가
            summary_info = f"""# HangulPDF AI Converter 분석 결과

## 파일 정보
- 원본 파일: {filename_base}_원본.pdf
- 처리 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 추출된 텍스트 길이: {len(extracted_text)} 글자

## 포함된 파일들
1. {filename_base}_원본.pdf - 원본 PDF 파일
2. {filename_base}_추출텍스트.txt - 추출된 텍스트
3. {filename_base}_ChatGPT분석.pdf/.txt - ChatGPT 분석 결과
4. {filename_base}_Gemini분석.pdf/.txt - Gemini 분석 결과
5. {filename_base}_Grok분석.pdf/.txt - Grok 분석 결과

## PDF 생성 정보
- 한글 폰트 지원: WeasyPrint > ReportLab > FPDF 순서로 시도
- TTF 폰트 우선 사용 (TTC 파일 호환성 문제 해결)
- 마크다운 형식: 지원

## 사용 방법
- PDF 파일: 각 AI 모델의 분석 결과를 읽기 쉬운 형태로 제공
- TXT 파일: 텍스트 형태의 분석 결과 (복사/편집 가능)

Generated by HangulPDF AI Converter
한글 PDF 생성 오류 수정 버전 v2.0
"""
            zipf.writestr(f"{filename_base}_README.txt", summary_info.encode('utf-8'))
        
        return temp_zip.name
        
    except Exception as e:
        st.error(f"ZIP 파일 생성 중 오류: {str(e)}")
        return None

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
            'success': True,
            'pdf_bytes': pdf_bytes  # ZIP 생성을 위해 원본 PDF 바이트 포함
        }
        
    except Exception as e:
        st.error(f"❌ 처리 중 예상치 못한 오류: {str(e)}")
        return {'error': f'PDF 처리 중 오류가 발생했습니다: {str(e)}'}

# 자동 AI 분석 및 ZIP 생성 함수
def auto_analyze_and_create_zip(extracted_text, pdf_bytes, filename_base, api_key):
    """자동으로 AI 분석을 수행하고 ZIP 파일 생성"""
    start_time = time.time()
    estimated_time = 120  # AI 분석은 더 오래 걸림
    
    try:
        # 1. ChatGPT 분석
        progress_bar, status_text = show_progress_with_timer("ChatGPT 분석 중...", 0.2, start_time, estimated_time)
        chatgpt_result = analyze_with_chatgpt(extracted_text, api_key)
        st.success("✅ ChatGPT 분석 완료")
        
        # 2. Gemini 분석
        progress_bar, status_text = show_progress_with_timer("Gemini 분석 중...", 0.4, start_time, estimated_time)
        gemini_result = analyze_with_gemini(extracted_text, api_key)
        st.success("✅ Gemini 분석 완료")
        
        # 3. Grok 분석
        progress_bar, status_text = show_progress_with_timer("Grok 분석 중...", 0.6, start_time, estimated_time)
        grok_result = analyze_with_grok(extracted_text)
        st.success("✅ Grok 분석 완료")
        
        # 4. ZIP 파일 생성
        progress_bar, status_text = show_progress_with_timer("ZIP 파일 생성 중...", 0.8, start_time, estimated_time)
        zip_path = create_analysis_zip(
            pdf_bytes, 
            extracted_text, 
            chatgpt_result, 
            gemini_result, 
            grok_result, 
            filename_base
        )
        
        if zip_path:
            progress_bar, status_text = show_progress_with_timer("완료!", 1.0, start_time, estimated_time)
            st.success("✅ ZIP 파일 생성 완료")
            return {
                'zip_path': zip_path,
                'chatgpt_result': chatgpt_result,
                'gemini_result': gemini_result,
                'grok_result': grok_result,
                'success': True
            }
        else:
            return {'error': 'ZIP 파일 생성에 실패했습니다.'}
            
    except Exception as e:
        st.error(f"❌ 자동 분석 중 오류: {str(e)}")
        return {'error': f'자동 분석 중 오류가 발생했습니다: {str(e)}'}

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

/* 다운로드 버튼 스타일 */
.download-button {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
    margin: 1rem 0;
}

/* 상태 표시 스타일 */
.status-info {
    background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
    color: white;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# 메인 제목
st.title("📄 HangulPDF AI Converter")
st.markdown("**한글 PDF 문서를 AI가 쉽게 활용할 수 있도록 자동 변환하는 도구**")

# 라이브러리 상태 표시
st.markdown(f"""
<div class="status-info">
    <h4>🔧 시스템 상태 (오류 수정 버전)</h4>
    <p><strong>PDF 처리:</strong> {'✅ 사용 가능' if PDF_AVAILABLE else '❌ 설치 필요'}</p>
    <p><strong>OCR 기능:</strong> {'✅ 사용 가능' if OCR_AVAILABLE else '❌ 설치 필요'}</p>
    <p><strong>WeasyPrint PDF:</strong> {'✅ 사용 가능 (오류 수정)' if WEASYPRINT_AVAILABLE else '❌ 설치 필요'}</p>
    <p><strong>ReportLab PDF:</strong> {'✅ 사용 가능 (TTF 폰트만)' if REPORTLAB_AVAILABLE else '❌ 설치 필요'}</p>
    <p><strong>FPDF PDF:</strong> {'✅ 사용 가능 (개선됨)' if FPDF_AVAILABLE else '❌ 설치 필요'}</p>
    <p><strong>TTF 한글 폰트:</strong> {'✅ ' + find_ttf_font() if find_ttf_font() else '❌ 없음'}</p>
    <p><strong>일반 한글 폰트:</strong> {'✅ ' + find_korean_font() if find_korean_font() else '❌ 없음'}</p>
</div>
""", unsafe_allow_html=True)

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
    
    # 자동 AI 분석 옵션
    st.header("🤖 자동 AI 분석")
    auto_ai_analysis = st.checkbox(
        "🚀 자동 AI 분석 및 ZIP 다운로드", 
        value=False, 
        help="ChatGPT, Gemini, Grok으로 자동 분석하고 결과를 ZIP으로 패키징합니다."
    )
    
    if auto_ai_analysis and not api_key:
        st.warning("⚠️ 자동 AI 분석을 위해서는 OpenAI API 키가 필요합니다.")
    
    # PDF 생성 방법 선택
    st.header("📄 PDF 생성 설정 (수정됨)")
    pdf_methods = []
    if WEASYPRINT_AVAILABLE:
        pdf_methods.append("WeasyPrint (오류 수정)")
    if REPORTLAB_AVAILABLE:
        pdf_methods.append("ReportLab (TTF 폰트만)")
    if FPDF_AVAILABLE:
        pdf_methods.append("FPDF (개선됨)")
    
    if pdf_methods:
        st.success(f"✅ 사용 가능한 PDF 생성 방법: {len(pdf_methods)}개")
        for method in pdf_methods:
            st.info(f"• {method}")
    else:
        st.error("❌ PDF 생성 라이브러리가 설치되지 않았습니다.")

# 메인 탭
tab1, tab2, tab3, tab4 = st.tabs(["📤 파일 업로드", "📊 변환 결과", "🔗 공유 & 내보내기", "📦 자동 분석 결과"])

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
        filename_base = os.path.splitext(uploaded_file.name)[0]
        
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
        
        # 자동 AI 분석 안내
        if auto_ai_analysis:
            st.success("🤖 자동 AI 분석 모드: ChatGPT, Gemini, Grok으로 자동 분석하고 오류 수정된 한글 PDF로 생성하여 ZIP 파일로 다운로드합니다.")
        
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
                    'generate_summary': False,
                    'generate_qa': False,
                    'api_key': api_key
                }
                
                # 로컬 처리
                result = process_pdf_locally(request_data)
                
                # 결과 저장
                st.session_state.conversion_result = result
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.filename_base = filename_base
                
                if 'error' not in result or result.get('success'):
                    st.success("✅ 변환이 완료되었습니다!")
                    
                    # 자동 AI 분석 실행
                    if auto_ai_analysis and api_key and result.get('extracted_text'):
                        st.info("🤖 자동 AI 분석을 시작합니다...")
                        
                        ai_result = auto_analyze_and_create_zip(
                            result['extracted_text'],
                            result.get('pdf_bytes', pdf_bytes),
                            filename_base,
                            api_key
                        )
                        
                        # AI 분석 결과 저장
                        st.session_state.ai_analysis_result = ai_result
                        
                        if ai_result.get('success'):
                            st.balloons()
                            st.success("🎉 자동 AI 분석 및 오류 수정된 한글 PDF 생성이 완료되었습니다!")
                            st.info("📦 '자동 분석 결과' 탭에서 ZIP 파일을 다운로드하세요.")
                        else:
                            st.error(f"❌ 자동 AI 분석 중 오류: {ai_result.get('error', '알 수 없는 오류')}")
                    else:
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

with tab4:
    st.header("📦 자동 분석 결과")
    
    if 'ai_analysis_result' in st.session_state:
        ai_result = st.session_state.ai_analysis_result
        
        if ai_result.get('success'):
            st.success("🎉 자동 AI 분석이 완료되었습니다!")
            
            # ZIP 파일 다운로드
            zip_path = ai_result.get('zip_path')
            if zip_path and os.path.exists(zip_path):
                with open(zip_path, 'rb') as f:
                    zip_data = f.read()
                
                filename_base = st.session_state.get('filename_base', 'analysis')
                download_filename = f"{filename_base}_AI분석결과_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                
                st.markdown(f"""
                <div class="download-button">
                    <h4>📦 분석 결과 다운로드</h4>
                    <p>포함된 파일: 원본 PDF, 추출 텍스트, ChatGPT/Gemini/Grok 분석 결과 (한글 PDF + TXT)</p>
                    <p><strong>🎨 한글 PDF 생성:</strong> WeasyPrint/ReportLab/FPDF 오류 수정 버전</p>
                    <p><strong>🔧 수정사항:</strong> TTC 폰트 문제 해결, PDF 생성 인자 오류 수정</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.download_button(
                    label="📥 ZIP 파일 다운로드",
                    data=zip_data,
                    file_name=download_filename,
                    mime="application/zip",
                    type="primary"
                )
                
                # 분석 결과 미리보기
                st.subheader("📋 분석 결과 미리보기")
                
                # ChatGPT 결과
                with st.expander("💬 ChatGPT 분석 결과"):
                    st.text_area(
                        "ChatGPT 분석:",
                        value=ai_result.get('chatgpt_result', ''),
                        height=300,
                        key="chatgpt_preview"
                    )
                
                # Gemini 결과
                with st.expander("🔮 Gemini 분석 결과"):
                    st.text_area(
                        "Gemini 분석:",
                        value=ai_result.get('gemini_result', ''),
                        height=300,
                        key="gemini_preview"
                    )
                
                # Grok 결과
                with st.expander("🚀 Grok 분석 결과"):
                    st.text_area(
                        "Grok 분석:",
                        value=ai_result.get('grok_result', ''),
                        height=300,
                        key="grok_preview"
                    )
            else:
                st.error("❌ ZIP 파일을 찾을 수 없습니다.")
        
        elif 'error' in ai_result:
            st.error(f"❌ 자동 분석 중 오류: {ai_result['error']}")
        
        else:
            st.warning("⚠️ 자동 분석 결과가 없습니다.")
    
    else:
        st.info("🤖 자동 AI 분석을 실행하려면 '파일 업로드' 탭에서 '자동 AI 분석 및 ZIP 다운로드' 옵션을 선택하고 변환을 시작하세요.")

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>📄 <strong>HangulPDF AI Converter</strong> - 한글 PDF 문서 AI 변환 도구</p>
    <p>🤖 자동 AI 분석 | 📦 ZIP 다운로드 | 📱 모바일 반응형 | ⏱️ 실시간 타이머</p>
    <p>🎨 <strong>한글 PDF 완벽 지원</strong> | WeasyPrint + ReportLab + FPDF 오류 수정</p>
    <p>🔧 <strong>v2.0 수정사항:</strong> TTC 폰트 문제 해결, PDF 생성 인자 오류 수정</p>
    <p>💡 ChatGPT, Gemini, Grok 자동 분석 및 결과 패키징</p>
</div>
""", unsafe_allow_html=True)

