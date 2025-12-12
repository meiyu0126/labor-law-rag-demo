import streamlit as st
import os
import shutil
import tempfile  # <--- 新增這個
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
st.title("⚖️ 企業勞基法智慧問答助手 (V8 - Tmp Dir Fix)")
st.caption("🚀 Powered by RAG (Database stored in /tmp for Write Permissions)")


# 2. 定義建立資料庫函式
def build_vector_db(file_path, db_path, embedding_function):
    try:
        print(f"--- [V8] 開始建立資料庫: {db_path} ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            st.error("❌ 錯誤: PDF 內容為空，請檢查 data/labor_law.pdf")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)
        st.write(f"📄 成功讀取 PDF，共切分出 `{len(chunks)}` 個片段，正在寫入暫存區資料庫...")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=db_path
        )
        print("✅ 資料庫建立成功！")
        return db
    except Exception as e:
        # 顯示更詳細的錯誤資訊
        st.error(f"❌ 資料庫建立失敗: {e}")
        return None


# 3. 載入 RAG 系統
def load_rag_system():
    load_dotenv()

    FILE_PATH = os.path.join("data", "labor_law.pdf")

    # 【關鍵修改】：使用系統暫存目錄 /tmp
    # 這樣可以保證有寫入權限，解決 readonly database 問題
    TMP_DIR = tempfile.gettempdir()
    CHROMA_PATH = os.path.join(TMP_DIR, "chroma_db_v8_tmp")

    # 顯示路徑給您看，確認是在 /tmp 底下
    print(f"Target Database Path: {CHROMA_PATH}")

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    # 強制刪除舊資料 (確保每次都是新的)
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except:
            pass

    # 執行建立
    with st.spinner(f"🏗️ [V8] 正在暫存區 ({CHROMA_PATH}) 重建資料庫... (約 20 秒)"):
        db = build_vector_db(FILE_PATH, CHROMA_PATH, embedding_function)

    if db is None:
        return None

    # --- RAG Chain 設定 ---
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


# 4. 初始化 Session
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手 (V8)。請輸入你想查詢的勞基法問題："}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 每次執行都載入系統
rag_chain = load_rag_system()

# 6. 處理使用者輸入
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
                                st.markdown(f"**來源 {i + 1}**: `{source}` (第 {page} 頁)")
                                st.text(doc.page_content[:100] + "...")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")