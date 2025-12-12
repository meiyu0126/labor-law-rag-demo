import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

# 1. 設定頁面 (V10 - Optimized)
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手 (V10 - High Precision)")
st.caption("🚀 Powered by RAG (Larger Chunks + More Context)")


# 2. 定義建立資料庫函式
def build_vector_db_in_memory(file_path, embedding_function):
    try:
        print(f"--- [V10] 開始建立記憶體資料庫 ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("❌ 錯誤: PDF 內容為空")
            return None

        # 【優化 1】加大 chunk_size，確保法條完整性
        # 原本 500 -> 改為 1000 (約包含 1-2 頁的完整內容，避免法條被切斷)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,  # 增加重疊，確保上下文連貫
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)
        print(f"📄 切分完成，共 {len(chunks)} 筆片段")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function
            # persist_directory=None (記憶體模式)
        )
        print("✅ 記憶體資料庫建立成功！")
        return db
    except Exception as e:
        print(f"❌ 建立失敗: {e}")
        return None


# 3. 載入 RAG 系統
@st.cache_resource(show_spinner=False)
def load_rag_system_v10():
    load_dotenv()

    FILE_PATH = os.path.join("data", "labor_law.pdf")
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    # 建立資料庫
    db = build_vector_db_in_memory(FILE_PATH, embedding_function)

    if db is None:
        return None

    # --- RAG Chain 設定 ---
    # 【優化 2】增加檢索數量 k
    # 原本 5 -> 改為 10，讓 AI 能參考更多相關條文 (如第30, 32條)
    retriever = db.as_retriever(search_kwargs={"k": 10})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    template = """你是一個專業的勞基法問答助手。
    請務必「只」依據以下的【參考資料】來回答使用者的問題。
    回答時，請優先引用具體的「法條條號」（例如：根據第 24 條...）。

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


# 4. 初始化 Session & 載入系統
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手 (V10)。請輸入你想查詢的勞基法問題："}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 呼叫載入 (外部轉圈圈)
if "rag_chain" not in st.session_state:
    with st.spinner("🚀 [V10] 系統升級中... 正在優化索引與切片 (約 20 秒)..."):
        st.session_state.rag_chain = load_rag_system_v10()

rag_chain = st.session_state.rag_chain

# 6. 處理輸入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在深度檢索法規資料庫..."):
                try:
                    result = rag_chain.invoke(prompt)
                    response_text = result["response"]
                    source_docs = result["context"]

                    st.write(response_text)

                    # 【優化 3】改善資料來源顯示
                    with st.expander("📚 查看資料來源 (Source Documents)"):
                        if not source_docs:
                            st.info("沒有找到相關的來源文件。")
                        else:
                            for i, doc in enumerate(source_docs):
                                # 嘗試將頁碼 +1 轉為人類可讀頁碼 (Python 是 0 開始，PDF 是 1 開始)
                                try:
                                    page_num = int(doc.metadata.get('page', 0)) + 1
                                except:
                                    page_num = doc.metadata.get('page', 'Unknown')

                                source = os.path.basename(doc.metadata.get('source', 'Unknown'))

                                # 標題顯示
                                st.markdown(f"**來源 {i + 1}**: `{source}` (第 {page_num} 頁)")

                                # 內容顯示：不截斷，顯示完整 Chunk 內容，並使用 Markdown 引用格式
                                st.markdown(f"> {doc.page_content}")
                                st.divider()

                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")
    else:
        st.error("系統初始化失敗，無法執行回答。")