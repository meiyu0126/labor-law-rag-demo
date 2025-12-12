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
st.title("⚖️ 企業勞基法智慧問答助手 (V17 - Large Stable)")
st.caption("🚀 Powered by text-embedding-3-large (Precision & Stability)")


# 2. 建立資料庫 (移除所有 st. 指令，只用 print)
def build_vector_db_in_memory(file_path, embedding_function):
    try:
        print(f"--- [V17] 開始建立記憶體資料庫 ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("❌ 錯誤: PDF 內容為空")
            return None

        # Large 模型：維持 500 字切片 + 100 重疊
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)
        print(f"📄 切分完成，共 {len(chunks)} 筆片段")

        # 指定唯一的 collection_name，避免維度衝突
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            collection_name="labor_laws_v17_large_final"
        )
        print("✅ 資料庫建立成功！")
        return db

    except Exception as e:
        print(f"❌ 建立失敗: {e}")
        return None


# 3. 載入系統 (使用快取，show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_rag_system_v17():
    load_dotenv()
    FILE_PATH = os.path.join("data", "labor_law.pdf")

    # 使用 Large 模型
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    db = build_vector_db_in_memory(FILE_PATH, embedding_function)
    if db is None: return None

    # 設定檢索器 (Large 模型通常很準，k=5 應該就夠抓到重點且排除雜訊)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    template = """你是一個專業的勞基法問答助手。
    請務必「只」依據以下的【參考資料】來回答使用者的問題。
    如果資料中沒有答案，請說「資料庫中找不到相關資訊」。

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
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！我是你的勞基法 AI 助手 (V17 - Large)。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 載入系統 (唯一的轉圈圈放在這裡)
if "rag_chain" not in st.session_state:
    with st.spinner("🚀 [V17] 系統初始化中... 正在啟用 Large 模型 (約 30 秒)..."):
        st.session_state.rag_chain = load_rag_system_v17()

rag_chain = st.session_state.rag_chain

# 6. 處理輸入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Large Model 深度檢索中..."):
                try:
                    result = rag_chain.invoke(prompt)
                    response_text = result["response"]
                    source_docs = result["context"]

                    st.write(response_text)

                    if source_docs:
                        with st.expander("📚 查看最佳參考來源 (Top 5)", expanded=True):
                            for i, doc in enumerate(source_docs):
                                try:
                                    # 嘗試修正頁碼顯示 (加 1)
                                    page_idx = doc.metadata.get('page', 0)
                                    page_num = int(page_idx) + 1
                                except:
                                    page_num = "?"

                                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                # 整理內文格式
                                content = doc.page_content.replace('\n', ' ')

                                st.markdown(f"### 🏅 來源 {i + 1}: 第 {page_num} 頁")
                                st.info(content)

                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")
    else:
        st.error("⚠️ 系統未成功初始化。")