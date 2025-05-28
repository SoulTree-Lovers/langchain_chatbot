import streamlit as st
import time
from typing import List, Dict, Any
import tempfile
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
from pdfminer.high_level import extract_text

load_dotenv()

# Custom CSS for modern UI
st.set_page_config(
    page_title="🤖 AI 기술 전문가 챗봇",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .stChatMessage[data-testid="stChatMessage"] {
        background-color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .document-list {
        margin: 1rem 0;
        padding: 1rem;
        background-color: white;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# handle streaming conversation
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def get_pdf_text(file):
    raw_text = extract_text(file)
    return raw_text

def process_uploaded_files(uploaded_files: List[Any]) -> Dict:
    if not uploaded_files:
        return None
    
    all_splits = []
    document_info = {}
    
    for uploaded_file in uploaded_files:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # PDF 텍스트 추출
            raw_text = get_pdf_text(tmp_file_path)
            
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            splits = text_splitter.create_documents([raw_text])
            
            # 문서 정보 저장
            document_info[uploaded_file.name] = {
                'splits': splits,
                'raw_text': raw_text
            }
            
            all_splits.extend(splits)
            
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_file_path)
    
    if all_splits:
        # FAISS 벡터 스토어 생성
        vectorstore = FAISS.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        return {
            'vectorstore': vectorstore,
            'document_info': document_info
        }
    return None

def generate_response(query_text: str, vectorstore, callback):
    # 더 많은 관련 문서 검색
    docs_list = vectorstore.similarity_search(query_text, k=5)
    docs = ""
    for i, doc in enumerate(docs_list):
        docs += f"문서 {i+1}:\n{doc.page_content}\n\n"
    
    # GPT-4 모델 사용
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        streaming=True,
        callbacks=[callback]
    )
    
    # 전문가 페르소나 설정
    rag_prompt = [
        SystemMessage(
            content="""당신은 AI 기술 전문가입니다. 
            주어진 여러 문서를 바탕으로 사용자의 질문에 전문적이고 상세하게 답변해주세요.
            여러 문서에서 관련 정보를 찾아 종합적으로 답변해주세요.
            문서에 정확한 정보가 없는 경우, '죄송합니다. 제공된 문서에서 해당 정보를 찾을 수 없습니다.'라고 답변하세요.
            답변은 항상 한국어로 해주세요."""
        ),
        HumanMessage(
            content=f"질문: {query_text}\n\n참고 문서:\n{docs}"
        ),
    ]

    response = llm(rag_prompt)
    return response.content

def generate_summarize(document_info: Dict, callback):
    llm = ChatOpenAI(
        model_name="gpt-4",
        temperature=0.7,
        streaming=True,
        callbacks=[callback]
    )
    
    summarize_prompt = PromptTemplate(
        input_variables=["text", "filename"],
        template="""다음 문서를 3-4개의 주요 포인트로 요약해주세요:
        문서명: {filename}
        
        내용:
        {text}
        
        요약은 다음 형식으로 작성해주세요:
        1. [주요 포인트 1]
        2. [주요 포인트 2]
        3. [주요 포인트 3]
        """
    )
    
    chain = LLMChain(llm=llm, prompt=summarize_prompt)
    
    summaries = []
    for filename, info in document_info.items():
        response = chain.run(text=info['raw_text'], filename=filename)
        summaries.append(f"📄 {filename}\n{response}\n")
    
    return "\n".join(summaries)

# UI 구성
st.title("🤖 AI 기술 전문가 챗봇")
st.markdown("""
이 챗봇은 여러 AI 기술 문서를 동시에 분석하고 설명해주는 전문가입니다.
PDF 문서들을 업로드하고 질문해보세요!
""")

# 사이드바
with st.sidebar:
    st.header("📄 문서 업로드")
    uploaded_files = st.file_uploader("PDF 파일들을 선택하세요", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("문서들을 처리중입니다..."):
            result = process_uploaded_files(uploaded_files)
            if result:
                st.session_state['vectorstore'] = result['vectorstore']
                st.session_state['document_info'] = result['document_info']
                st.success("모든 문서가 성공적으로 처리되었습니다!")
                
                # 업로드된 문서 목록 표시
                st.markdown("### 📚 업로드된 문서 목록")
                for filename in result['document_info'].keys():
                    st.markdown(f"- {filename}")

# 채팅 인터페이스
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(
            role="assistant",
            content="안녕하세요! 저는 AI 기술 전문가 챗봇입니다. PDF 문서들을 업로드하고 AI 기술에 대해 어떤 것이든 물어보세요!"
        )
    ]

# 채팅 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg.role):
        st.write(msg.content)

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        
        if prompt.lower() == "요약":
            if 'document_info' in st.session_state:
                response = generate_summarize(st.session_state['document_info'], stream_handler)
            else:
                response = "문서가 업로드되지 않았습니다. 먼저 PDF 파일을 업로드해주세요."
        else:
            if 'vectorstore' in st.session_state:
                response = generate_response(prompt, st.session_state['vectorstore'], stream_handler)
            else:
                response = "문서가 업로드되지 않았습니다. 먼저 PDF 파일을 업로드해주세요."
        
        st.session_state["messages"].append(
            ChatMessage(role="assistant", content=response)
        ) 