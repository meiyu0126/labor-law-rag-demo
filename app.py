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
import tempfile  # <--- 新增這個模組來處理上傳檔案

# 1. 設定頁面
st.set_page_config(page_title="企業智能問答助手", page_icon="📂")
st.title("📂 企業智能文件問答助手 (V22 - Upload Support)")
st.caption("🚀 Powered by Large Model + Custom PDF Upload")

# --- 側邊欄：檔案上傳區 ---
with st.sidebar:
    st.header("📂 文件上傳")
    uploaded_file = st.file_uploader("請上傳您的 PDF 文件", type=["pdf"])

    st.divider()
    st.header("⚙️ 系統參數")
    st.info(f"Chunk Size: 1000")
    st.info(f"Chunk Overlap: 30")

    if uploaded_file:
        st.success(f"目前使用文件：\n{uploaded_file.name}")
    else:
        st.warning("目前使用預設文件：\n勞動基準法.pdf")


# -------------------------

# 2. 建立資料庫 (雲端安全版 - In-Memory)
def build_vector_db_in_memory(file_path, embedding_function):
    try:
        # 顯示處理中的檔案名稱
        file_name = os.path.basename(file_path)
        print(f"--- [V22] 開始處理檔案: {file_name} ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("❌ 錯誤: PDF 內容為空")
            return None

        # 切分設定 (維持您的最佳參數)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)

        # 過濾雜訊
        clean_chunks = [c for c in chunks if len(c.page_content) > 150]

        print(f"📄 切分完成，共 {len(clean_chunks)} 筆有效片段")

        # 使用檔案名稱來作為 Collection Name，確保不同檔案不會混在一起
        # 這裡做一點字串處理，把檔名變成合法的 Collection Name (只留英數)
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', file_name)[:50]
        collection_name = f"rag_coll_{safe_name}"

        db = Chroma.from_documents(
            documents=clean_chunks,
            embedding=embedding_function,
            collection_name=collection_name
        )
        print("✅ 資料庫建立成功 (記憶體模式)！")
        return db

    except Exception as e:
        print(f"❌ 建立失敗: {e}")
        return None


# 3. 載入系統 (快取邏輯調整)
# 這裡我們把 file_path 當作快取的 key
# 只要 file_path 改變 (例如使用者上傳了新檔案)，快取就會失效，自動重建 DB
@st.cache_resource(show_spinner=False)
def load_rag_system_v22(target_file_path):
    load_dotenv()

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    db = build_vector_db_in_memory(target_file_path, embedding_function)
    if db is None: return None

    # 維持您的 MMR 設定
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.85
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


# 4. 處理檔案邏輯 (關鍵步驟)
if uploaded_file:
    # 如果使用者有上傳檔案
    # 1. 建立一個暫存檔
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
else:
    # 如果沒上傳，使用預設的勞基法
    tmp_file_path = os.path.join("data", "labor_law.pdf")

# 5. 初始化 Session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！請上傳 PDF 文件，或直接詢問勞基法相關問題。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. 載入系統 (根據 tmp_file_path 決定是否重建)
if "rag_chain" not in st.session_state or st.session_state.get("current_file") != tmp_file_path:
    with st.spinner("🚀 正在分析文件並建立知識庫..."):
        # 呼叫建庫函式
        chain = load_rag_system_v22(tmp_file_path)
        # 將 chain 存入 session
        st.session_state.rag_chain = chain
        # 記錄目前使用的檔案路徑，以便偵測變更
        st.session_state.current_file = tmp_file_path

        # 如果是切換檔案，建議清空對話紀錄，避免混淆 (可選)
        # st.session_state.messages = [{"role": "assistant", "content": "已切換文件，請發問！"}]

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
                        with st.expander("📚 查看最佳參考來源", expanded=True):
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