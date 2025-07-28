import streamlit as st
import requests
import json
import os
from io import BytesIO
import base64

# 로컬 PDF 처리 함수 (상단으로 이동)
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
        
        return result
        
    except Exception as e:
        return {"error": f"PDF 처리 중 오류: {str(e)}"}

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
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password", 
    value=st.secrets.get("OPENAI_API_KEY", ""),
    help="sk-로 시작하는 OpenAI API 키를 입력하세요"
)

if not openai_api_key:
    st.warning("⚠️ OpenAI API 키를 입력해주세요. (요약 및 Q&A 생성 기능을 사용하려면 필요합니다)")
    st.info("💡 텍스트 추출 기능은 API 키 없이도 사용 가능합니다.")

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
            generate_summary = st.checkbox("요약 생성", value=True, disabled=not openai_api_key)
            
        with col2:
            generate_qa = st.checkbox("질문-답변 생성", value=False, disabled=not openai_api_key)
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
        
        # 오류 처리
        if 'error' in result:
            st.error(f"❌ {result['error']}")
        elif 'api_error' in result:
            st.error(f"❌ {result['api_error']}")
        else:
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
        
        if 'error' not in result and 'api_error' not in result:
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
                    chatgpt_prompt = f"""다음 문서를 분석해주세요:

{result['extracted_text']}

위 문서에 대해 다음을 수행해주세요:
1. 주요 내용 요약
2. 핵심 키워드 추출
3. 중요한 질문 3개 생성"""
                    
                    st.text_area(
                        "ChatGPT에 복사하여 사용하세요:", 
                        value=chatgpt_prompt, 
                        height=200,
                        key="chatgpt_prompt"
                    )
                    
                    # Gemini 프롬프트
                    st.markdown("**🔮 Gemini 프롬프트:**")
                    gemini_prompt = f"""문서 분석 요청:

{result['extracted_text']}

분석 항목:
- 문서 유형 및 목적 파악
- 핵심 내용 정리
- 실행 가능한 액션 아이템 추출
- 관련 질문 생성"""
                    
                    st.text_area(
                        "Gemini에 복사하여 사용하세요:", 
                        value=gemini_prompt, 
                        height=150,
                        key="gemini_prompt"
                    )
                    
                    # Grok 프롬프트
                    st.markdown("**🚀 Grok 프롬프트:**")
                    grok_prompt = f"""Hey Grok, analyze this Korean document:

{result['extracted_text']}

Please provide:
- Document summary in Korean
- Key insights and takeaways
- Potential follow-up questions
- Creative perspectives on the content"""
                    
                    st.text_area(
                        "Grok에 복사하여 사용하세요:", 
                        value=grok_prompt, 
                        height=150,
                        key="grok_prompt"
                    )
                    
                    # 복사 도움말
                    st.info("💡 각 텍스트 박스를 클릭하고 Ctrl+A → Ctrl+C로 전체 내용을 복사할 수 있습니다.")
                
        else:
            st.error("변환 결과에 오류가 있어 내보내기를 할 수 없습니다.")
    else:
        st.info("📤 먼저 PDF 파일을 업로드하고 변환해주세요.")

# 푸터
st.markdown("---")
st.markdown("🔧 **HangulPDF AI Converter** | 한글 PDF 문서 AI 변환 도구")

