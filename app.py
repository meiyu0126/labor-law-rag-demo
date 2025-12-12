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

# 1. 設定頁面
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手 (V13 - Final Clean)")
st.caption("🚀 Powered by RAG (Precision Tuned: k=5, Threshold=0.5)")


# 2. 建立資料庫 (純邏輯)
def build_vector_db_in_memory(file_path, embedding_function):
    try:
        print(f"--- [V13] 開始建立記憶體資料庫 ---")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs: return None

        # 維持 V12 的切片策略：500字 + 200重疊 (保證法條完整性)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)

        db = Chroma.from_documents(documents=chunks, embedding=embedding_function)
        print("✅ 記憶體資料庫建立成功！")
        return db
    except Exception as e:
        print(f"❌ 建立失敗: {e}")
        return None


# 3. 載入系統
@st.cache_resource(show_spinner=False)
def load_rag_system_v13():
    load_dotenv()
    FILE_PATH = os.path.join("data", "labor_law.pdf")
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    db = build_vector_db_in_memory(FILE_PATH, embedding_function)
    if db is None: return None

    # 【關鍵優化】：提高門檻，減少數量
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,  # 提高門檻到 0.5 (過濾掉退休金那些似是而非的條文)
            "k": 5  # 只抓前 5 名 (剛好涵蓋完整的第 24 條相關 Chunk，切掉第 6 名的雜訊)
        }
    )

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


# 4. 初始化 Session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是你的勞基法 AI 助手 (V13)。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 載入系統
if "rag_chain" not in st.session_state:
    with st.spinner("🚀 [V13] 系統微調中... 正在優化檢索精度..."):
        st.session_state.rag_chain = load_rag_system_v13()

rag_chain = st.session_state.rag_chain

# 6. 處理輸入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在檢索最相關法條..."):
                try:
                    result = rag_chain.invoke(prompt)
                    response_text = result["response"]
                    source_docs = result["context"]

                    st.write(response_text)

                    # 顯示資料來源 (只顯示通過門檻的)
                    if source_docs:
                        with st.expander("📚 查看最佳參考來源 (Filtered Sources)", expanded=False):
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
                    else:
                        st.warning("⚠️ 查無高相關性的法規條文 (可能因相似度低於 0.5 門檻而被過濾)。")

                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    if "No relevant" in str(e) or "empty" in str(e):
                        st.warning("⚠️ 查無相關法規，請嘗試換個問法。")
                    else:
                        st.error(f"發生錯誤：{e}")
    else:
        st.error("系統初始化失敗。")