import streamlit as st
import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 1. 設定頁面
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手")
st.caption("🚀 Powered by RAG (Auto-Build on Cloud)")


# 2. 定義一個函式來「現場建立」資料庫
def build_vector_db(file_path, db_path, embedding_function):
    # 確保這行文字存在，這樣你才會在網頁上看到轉圈圈
    with st.spinner("🏗️ 偵測到新環境！正在重新建立向量資料庫 (約需 20 秒)..."):
        # 讀取 PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 切分文字
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)

        # 建立資料庫
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=db_path
        )
        return db


# 3. 載入 RAG 系統 (快取資源)
@st.cache_resource
def load_rag_system():
    load_dotenv()

    # 設定路徑 (改個新名字，避免讀到舊的壞檔)
    FILE_PATH = os.path.join("data", "labor_law.pdf")
    CHROMA_PATH = "chroma_db_v3_force_rebuild"

    # 準備 Embedding 模型
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    # --- 關鍵邏輯：檢查資料庫是否存在 ---
    if os.path.exists(CHROMA_PATH):
        # 嘗試讀取
        try:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
            # 簡單測試是否能運作，如果報錯就重建
            db._collection.count()
        except:
            # 如果讀取失敗 (例如 Windows/Linux 相容性問題)，刪除重建
            shutil.rmtree(CHROMA_PATH)
            db = build_vector_db(FILE_PATH, CHROMA_PATH, embedding_function)
    else:
        # 如果不存在，直接建立
        db = build_vector_db(FILE_PATH, CHROMA_PATH, embedding_function)

    # --- 以下是正常的 RAG 流程 ---

    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    template = """你是一個專業的勞基法問答助手。
    請依據以下的【參考資料】來回答使用者的問題。
    如果資料中沒有答案，請直接說「抱歉，根據目前的資料庫，我無法回答這個問題」。

    【參考資料】：
    {context}

    使用者問題：{question}

    回答："""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retrieval_step = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    answer_step = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
    )

    final_chain = retrieval_step | RunnableParallel({
        "response": answer_step,
        "context": lambda x: x["context"]
    })

    return final_chain


rag_chain = load_rag_system()

# 4. 初始化對話
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手。請輸入你想查詢的勞基法問題："}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 處理輸入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在檢索法規資料庫..."):
                try:
                    result = rag_chain.invoke(prompt)
                    response_text = result["response"]
                    source_docs = result["context"]

                    st.write(response_text)

                    with st.expander("📚 查看資料來源 (Source Documents)"):
                        if not source_docs:
                            st.info("沒有找到相關的來源文件。")
                        else:
                            for i, doc in enumerate(source_docs):
                                page = doc.metadata.get('page', 'Unknown')
                                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                # [重要] 這裡加上來源驗證
                                st.markdown(f"**來源 {i + 1}**: `{source}` (第 {page} 頁)")
                                st.text(doc.page_content[:100] + "...")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")