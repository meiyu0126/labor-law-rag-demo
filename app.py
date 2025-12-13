import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from operator import itemgetter
import os

# 1. 設定頁面
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手")
st.caption("🚀 Powered by Large Model ")


# 2. 建立或載入資料庫 (Persistence 版本)
def build_vector_db_in_memory(file_path, embedding_function):
    # 設定資料庫要存在哪個資料夾 (請確保這個資料夾名稱有在 .gitignore 裡)
    PERSIST_DIR = "chroma_db_data"

    # 檢查資料夾是否存在
    if os.path.exists(PERSIST_DIR):
        print(f"--- [V19] 發現已存在的資料庫 ({PERSIST_DIR})，直接載入，不扣款！ ---")
        # 直接讀取硬碟上的資料庫
        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_function,
            collection_name="labor_laws_v19_optimized"
        )
        return db

    # 如果資料夾不存在，才開始建立
    try:
        print(f"--- [V19] 找不到資料庫，開始建立新資料庫 (會呼叫 OpenAI API)... ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("❌ 錯誤: PDF 內容為空")
            return None

        # 切分設定
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=20,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)

        # 過濾雜訊
        clean_chunks = [c for c in chunks if len(c.page_content) > 150]

        # 建立資料庫並指定儲存路徑 (persist_directory)
        db = Chroma.from_documents(
            documents=clean_chunks,
            embedding=embedding_function,
            collection_name="labor_laws_v19_optimized",
            persist_directory=PERSIST_DIR  # <--- 關鍵：告訴它存到硬碟
        )
        print("✅ 資料庫建立並儲存成功！")
        return db

    except Exception as e:
        print(f"❌ 建立失敗: {e}")
        return None

# --- 建議把 format_chat_history 搬到這裡 ---
def format_chat_history(messages):
    history_text = ""
    recent_messages = messages[-6:]
    for msg in recent_messages:
        if msg["role"] == "user":
            history_text += f"使用者: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"助手: {msg['content']}\n"
    return history_text

# 3. 載入系統
@st.cache_resource(show_spinner=False)
def load_rag_system_v19():
    load_dotenv()
    FILE_PATH = os.path.join("data", "labor_law.pdf")

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    db = build_vector_db_in_memory(FILE_PATH, embedding_function)
    if db is None: return None

    # 【優化 2 & 3】調整 MMR 參數
    # lambda_mult=0.85: 強烈要求「相關性」，只允許一點點「多樣性」。
    # k=4: 只取前 4 名，避免第 5 名開始出現不相關的雜訊。
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.85
        }
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    #template = """你是一個專業的勞基法問答助手。
    #請務必「只」依據以下的【參考資料】來回答使用者的問題。

    #【參考資料】：
    #{context}

    #使用者問題：{question}

    #回答："""
    # 修改後的 template (加入 {chat_history})
    template = """你是一個專業的勞基法問答助手。
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

    # 修改 Chain 的輸入處理
    # 這裡的意思是：
    # 1. context: 拿字典裡的 "question" 去做檢索 (retriever)
    # 2. question: 拿字典裡的 "question" 直接傳下去
    # 3. chat_history: 拿字典裡的 "chat_history" 直接傳下去
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


# 4. 初始化 Session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是你的勞基法 AI 助手,請輸入勞基法相關查詢我會盡力為你提供說明。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 載入系統
if "rag_chain" not in st.session_state:
    with st.spinner("🚀 [V19] 系統優化中... 正在調整切片大小與權重..."):
        st.session_state.rag_chain = load_rag_system_v19()

rag_chain = st.session_state.rag_chain

# 6. 處理輸入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在進行精準檢索..."):
                try:
                    # ---【關鍵修改開始】---

                    # 1. 整理歷史紀錄
                    history_str = format_chat_history(st.session_state.messages[:-1])  # 排除剛剛輸入的那句

                    # 2. 改成傳入「字典」，包含問題與歷史
                    result = rag_chain.invoke({
                        "question": prompt,
                        "chat_history": history_str
                    })

                    # ---【關鍵修改結束】---

                    response_text = result["response"]
                    source_docs = result["context"]
                    st.write(response_text)

                    if source_docs:
                        with st.expander("📚 查看最佳參考來源 (Top 4 - Optimized)", expanded=True):
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