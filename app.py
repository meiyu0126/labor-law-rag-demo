import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# 1. 載入環境變數
load_dotenv()

# 設定網頁標題
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手")
st.caption("🚀 Powered by RAG (LangChain + ChromaDB + OpenAI)")


# 2. 載入環境與資料庫 (利用 cache resource 加速)
@st.cache_resource
def load_rag_system():
    CHROMA_PATH = "chroma_db"

    # 檢查資料庫是否存在
    if not os.path.exists(CHROMA_PATH):
        st.error("❌ 找不到向量資料庫，請先執行 ingest.py 建立資料庫！")
        return None

    # 準備 Embedding 模型 (請確認這裡跟您 ingest.py 用的模型名稱一致)
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    # 載入資料庫
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 【回歸原始設定】：只設定 k=5，不加任何過濾門檻
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 設定 LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 建立標準的問答鏈 (RetrievalQA)
    # 這是 LangChain 封裝好的標準流程，它會自動把檢索到的文字塞給 LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True  # 雖然這裡設 True，但我們在介面上選擇不顯示它
    )

    return qa_chain


# 初始化系統
qa_chain = load_rag_system()

# 3. 聊天介面
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手。請問關於加班費、休假或工時，有什麼想問的嗎？"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    # 顯示使用者輸入
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if qa_chain:
        # 取得 AI 回答
        with st.chat_message("assistant"):
            with st.spinner("正在檢索勞基法規..."):
                # 呼叫 QA Chain
                response = qa_chain.invoke({"query": prompt})
                result = response["result"]

                # 顯示回答
                st.write(result)

                # 更新對話紀錄
                st.session_state.messages.append({"role": "assistant", "content": result})

                # 【註】：這裡故意不寫出 source documents 的程式碼
                # 這樣就恢復到了您說「原本正確」時的狀態