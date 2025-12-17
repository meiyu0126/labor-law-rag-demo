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
import tempfile
import time

# 1. 設定頁面
st.set_page_config(page_title="企業智能問答助手", page_icon="📂")
st.title("📂 企業智能文件問答助手 (V24 - Precision Focus)")
st.caption("🚀 Powered by Large Model + Strict Filtering")

# --- 側邊欄 ---
with st.sidebar:
    st.header("📂 文件上傳")
    uploaded_file = st.file_uploader("請上傳您的 PDF 文件", type=["pdf"])

    st.divider()
    st.header("⚙️ 系統參數")
    st.info(f"Chunk Size: 600")
    st.info(f"Chunk Overlap: 30")
    st.info(f"Top-K: 3 (Strict)") # 顯示目前的設定

    if uploaded_file:
        st.success(f"目前使用文件：\n{uploaded_file.name}")
    else:
        st.warning("目前使用預設文件：\n勞動基準法.pdf")
# -------------------------

# 2. 建立資料庫
def build_vector_db_in_memory(file_path, embedding_function):
    try:
        file_name = os.path.basename(file_path)
        print(f"--- [V24] 開始處理檔案: {file_name} ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("❌ 錯誤: PDF 內容為空")
            return None

        # 切分設定
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)

        # 過濾雜訊
        clean_chunks = [c for c in chunks if len(c.page_content) > 150]

        print(f"📄 切分完成，共 {len(clean_chunks)} 筆有效片段")

        # 產生唯一 ID
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', file_name)[:30]
        unique_id = int(time.time())
        collection_name = f"rag_{safe_name}_{unique_id}"

        db = Chroma.from_documents(
            documents=clean_chunks,
            embedding=embedding_function,
            collection_name=collection_name
        )
        print(f"✅ 資料庫建立成功 (ID: {unique_id})！")
        return db

    except Exception as e:
        print(f"❌ 建立失敗: {e}")
        return None


# 3. 載入系統
@st.cache_resource(show_spinner=False)
def load_rag_system_v24(target_file_path):
    load_dotenv()

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    db = build_vector_db_in_memory(target_file_path, embedding_function)
    if db is None: return None

    # 【關鍵修改】
    # 1. k=3: 只取前 3 名，砍掉第 4 名以後的雜訊。
    # 2. lambda_mult=0.7: 稍微調高相似度權重，減少因為「追求多樣」而抓到退休金的情況。
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 20,
            "lambda_mult": 0.8
        }
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    template = """你是一個專業的文件問答助手。
    請依據【參考資料】與【歷史對話】來回答使用者的問題。

    【歷史對話】：
    {chat_history}

    【參考資料】：
    {context}

    使用者問題：{question}

    回答："""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    from operator import itemgetter

    retrieval_step = RunnableParallel(
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
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


# --- 歷史訊息處理 ---
def format_chat_history(messages):
    history_text = ""
    recent_messages = messages[-6:]
    for msg in recent_messages:
        if msg["role"] == "user":
            history_text += f"使用者: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"助手: {msg['content']}\n"
    return history_text


# 4. 處理檔案邏輯
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
else:
    tmp_file_path = os.path.join("data", "labor_law.pdf")

# 5. 初始化 Session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！請上傳 PDF 文件，或直接詢問勞基法相關問題。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. 載入系統
if "rag_chain" not in st.session_state or st.session_state.get("current_file") != tmp_file_path:
    with st.spinner("🚀 正在優化檢索模型..."):
        chain = load_rag_system_v24(tmp_file_path)
        st.session_state.rag_chain = chain
        st.session_state.current_file = tmp_file_path

rag_chain = st.session_state.rag_chain

# 7. 處理輸入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在檢索..."):
                try:
                    history_str = format_chat_history(st.session_state.messages[:-1])

                    result = rag_chain.invoke({
                        "question": prompt,
                        "chat_history": history_str
                    })

                    response_text = result["response"]
                    source_docs = result["context"]
                    st.write(response_text)

                    if source_docs:
                        with st.expander("📚 查看最佳參考來源 (Top 3)", expanded=True):
                            for i, doc in enumerate(source_docs):
                                try:
                                    page_idx = doc.metadata.get('page', 0)
                                    page_num = int(page_idx) + 1
                                except:
                                    page_num = "?"

                                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                content = doc.page_content.replace('\n', ' ')

                                st.markdown(f"### 🏅 來源 {i + 1}: 第 {page_num} 頁")
                                st.info(content)

                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")
    else:
        st.error("系統初始化失敗。")