import streamlit as st
import requests
import json
import os
from io import BytesIO
import base64

# 페이지 설정
st.set_page_config(
    page_title="HangulPDF AI Converter",
    page_icon="📄",
    layout="wide"
)

# 제목
st.title("📄 HangulPDF AI Converter")
st.markdown("한글 PDF 문서를 AI가 활용하기 쉬운 형태로 변환하고 분석하는 도구입니다.")

# API 서버 URL (로컬 개발용)
API_BASE_URL = "http://localhost:8000"

# 사이드바 - API 키 설정
st.sidebar.header("🔑 API 설정")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=st.secrets.get("OPENAI_API_KEY", ""))

if not openai_api_key:
    st.warning("⚠️ OpenAI API 키를 입력해주세요.")
    st.stop()

# 메인 컨텐츠
tab1, tab2, tab3 = st.tabs(["📄 PDF 업로드 & 변환", "📊 분석 결과", "🔗 공유 & 내보내기"])

with tab1:
    st.header("PDF 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "PDF 파일을 선택하세요",
        type=['pdf'],
        help="한글로 작성된 PDF 문서를 업로드하세요."
    )
    
    if uploaded_file is not None:
        st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}")
        
        # 파일 정보 표시
        file_details = {
            "파일명": uploaded_file.name,
            "파일 크기": f"{len(uploaded_file.getvalue())} bytes",
            "파일 타입": uploaded_file.type
        }
        st.json(file_details)
        
        # 변환 옵션
        st.subheader("🔧 변환 옵션")
        col1, col2 = st.columns(2)
        
        with col1:
            extract_text = st.checkbox("텍스트 추출", value=True)
            generate_summary = st.checkbox("요약 생성", value=True)
            
        with col2:
            generate_qa = st.checkbox("질문-답변 생성", value=False)
            clean_text = st.checkbox("텍스트 정제", value=True)
        
        # 변환 실행
        if st.button("🚀 변환 시작", type="primary"):
            with st.spinner("PDF를 분석하고 변환하는 중..."):
                try:
                    # 파일을 base64로 인코딩
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
                    st.session_state.conversion_result = process_pdf_locally(request_data)
                    st.success("✅ 변환이 완료되었습니다!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 변환 중 오류가 발생했습니다: {str(e)}")

with tab2:
    st.header("📊 분석 결과")
    
    if 'conversion_result' in st.session_state:
        result = st.session_state.conversion_result
        
        # 텍스트 추출 결과
        if 'extracted_text' in result:
            st.subheader("📝 추출된 텍스트")
            with st.expander("전체 텍스트 보기"):
                st.text_area("", value=result['extracted_text'], height=300)
        
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 텍스트 내보내기")
            if 'extracted_text' in result:
                st.download_button(
                    label="📄 텍스트 파일 다운로드",
                    data=result['extracted_text'],
                    file_name=f"{uploaded_file.name}_extracted.txt",
                    mime="text/plain"
                )
            
            if 'summary' in result:
                st.download_button(
                    label="📋 요약 파일 다운로드",
                    data=result['summary'],
                    file_name=f"{uploaded_file.name}_summary.txt",
                    mime="text/plain"
                )
        
        with col2:
            st.subheader("🤖 AI 모델 연동")
            st.markdown("**ChatGPT 프롬프트:**")
            if 'extracted_text' in result:
                chatgpt_prompt = f"다음 문서를 분석해주세요:\n\n{result['extracted_text'][:1000]}..."
                st.text_area("", value=chatgpt_prompt, height=150)
            
            st.markdown("**Gemini/Grok 연동:**")
            st.info("추출된 텍스트를 복사하여 다른 AI 모델에 직접 입력할 수 있습니다.")
    else:
        st.info("📤 먼저 PDF 파일을 업로드하고 변환해주세요.")

# 로컬 PDF 처리 함수
def process_pdf_locally(request_data):
    """PDF를 로컬에서 직접 처리하는 함수"""
    try:
        import pdfplumber
        import openai
        
        # base64 디코딩
        file_content = base64.b64decode(request_data['file_content'])
        
        # PDF 텍스트 추출
        extracted_text = ""
        with pdfplumber.open(BytesIO(file_content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
        
        result = {"extracted_text": extracted_text}
        
        # OpenAI 클라이언트 설정
        if request_data.get('openai_api_key'):
            client = openai.OpenAI(api_key=request_data['openai_api_key'])
            
            # 요약 생성
            if request_data['options'].get('generate_summary'):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "당신은 한국어 문서 요약 전문가입니다. 주요 내용을 간결하게 정리해주세요."},
                            {"role": "user", "content": f"다음 문서를 요약해주세요:\n\n{extracted_text[:3000]}"}
                        ],
                        max_tokens=500
                    )
                    result['summary'] = response.choices[0].message.content
                except Exception as e:
                    result['summary'] = f"요약 생성 중 오류: {str(e)}"
            
            # 질문-답변 생성
            if request_data['options'].get('generate_qa'):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "문서 내용을 바탕으로 3개의 질문과 답변을 생성해주세요."},
                            {"role": "user", "content": f"다음 문서에서 중요한 질문 3개와 답변을 만들어주세요:\n\n{extracted_text[:2000]}"}
                        ],
                        max_tokens=800
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
                    result['qa_pairs'] = [{"question": "질문 생성 오류", "answer": str(e)}]
        
        return result
        
    except Exception as e:
        return {"error": f"PDF 처리 중 오류: {str(e)}"}

# 푸터
st.markdown("---")
st.markdown("🔧 **HangulPDF AI Converter** | 한글 PDF 문서 AI 변환 도구")

