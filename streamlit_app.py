import streamlit as st
import requests
import json
import os
from io import BytesIO
import base64
import streamlit.components.v1 as components

# 복사 버튼 HTML/JavaScript 함수
def create_copy_button(text_content, button_id):
    """복사 버튼을 생성하는 함수"""
    copy_button_html = f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <button id="{button_id}" onclick="copyToClipboard_{button_id}()" 
                style="background-color: #ff4b4b; color: white; border: none; 
                       padding: 5px 10px; border-radius: 5px; cursor: pointer; 
                       font-size: 12px; margin-left: 10px;">
            📋 복사하기
        </button>
    </div>
    <script>
    function copyToClipboard_{button_id}() {{
        const text = `{text_content.replace('`', '\\`').replace('$', '\\$')}`;
        navigator.clipboard.writeText(text).then(function() {{
            document.getElementById('{button_id}').innerHTML = '✅ 복사됨!';
            document.getElementById('{button_id}').style.backgroundColor = '#00cc44';
            setTimeout(function() {{
                document.getElementById('{button_id}').innerHTML = '📋 복사하기';
                document.getElementById('{button_id}').style.backgroundColor = '#ff4b4b';
            }}, 2000);
        }}, function(err) {{
            alert('복사 실패: ' + err);
        }});
    }}
    </script>
    """
    return copy_button_html

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
                    
                    # 복사 버튼
                    components.html(create_copy_button(chatgpt_prompt, "chatgpt_copy"), height=50)
                    
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
                    
                    # 복사 버튼
                    components.html(create_copy_button(gemini_prompt, "gemini_copy"), height=50)
                    
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
                    
                    # 복사 버튼
                    components.html(create_copy_button(grok_prompt, "grok_copy"), height=50)
                    
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

