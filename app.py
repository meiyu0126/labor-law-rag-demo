import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1. 設定頁面標題
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手")
st.caption("🚀 Powered by RAG (LangChain + ChromaDB + OpenAI)")


# 2. 載入環境與資料庫 (利用 cache resource 加速，不用每次重新讀取)
@st.cache_resource
def load_rag_system():
    load_dotenv()
    CHROMA_PATH = "chroma_db"

    # 檢查資料庫是否存在
    if not os.path.exists(CHROMA_PATH):
        st.error("❌ 找不到向量資料庫，請先執行 ingest.py 建立資料庫！")
        return None

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # 這裡我們維持 k=5 的成功設定
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    template = """你是一個專業的勞基法問答助手。
    請依據以下的【參考資料】來回答使用者的問題。
    如果資料中沒有答案，請直接說「抱歉，根據目前的資料庫，我無法回答這個問題」，不要試圖憑空捏造。

    【參考資料】：
    {context}

    使用者問題：{question}

    回答："""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


# 初始化 RAG 鏈
rag_chain = load_rag_system()

# 3. 處理對話歷史 (Session State)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手。請問關於加班費、休假或工時，有什麼想問的嗎？"}]

# 顯示歷史訊息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 4. 處理使用者輸入
if prompt := st.chat_input():
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 生成 AI 回應
    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在檢索法規資料庫..."):
                response = rag_chain.invoke(prompt)
                st.write(response)

        # 存入歷史紀錄
        st.session_state.messages.append({"role": "assistant", "content": response})