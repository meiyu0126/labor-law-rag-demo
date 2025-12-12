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

# 1. 設定頁面 (V11 - Clean UX)
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手 (V11 - Best UX)")
st.caption("🚀 Powered by RAG (High Precision & Clean Display)")


# 2. 定義建立資料庫函式 (純邏輯)
def build_vector_db_in_memory(file_path, embedding_function):
    try:
        print(f"--- [V11] 開始建立記憶體資料庫 ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("❌ 錯誤: PDF 內容為空")
            return None

        # 【優化 1】縮小 chunk_size 回到 500，提升精準度
        # 這樣可以避免把不相關的法條（如產假）跟加班費混在一起
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 改回較小的切片，讓向量更精確
            chunk_overlap=150,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)
        print(f"📄 切分完成，共 {len(chunks)} 筆片段")

        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function
        )
        print("✅ 記憶體資料庫建立成功！")
        return db
    except Exception as e:
        print(f"❌ 建立失敗: {e}")
        return None


# 3. 載入 RAG 系統 (使用快取)
@st.cache_resource(show_spinner=False)
def load_rag_system_v11():
    load_dotenv()

    FILE_PATH = os.path.join("data", "labor_law.pdf")
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    db = build_vector_db_in_memory(FILE_PATH, embedding_function)

    if db is None:
        return None

    # 【優化 2】AI 讀取 10 筆，確保知識完整
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
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手 (V11)。請輸入你想查詢的勞基法問題："}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 呼叫載入
if "rag_chain" not in st.session_state:
    with st.spinner("🚀 系統啟動中... 正在記憶體中構建高精度知識庫..."):
        st.session_state.rag_chain = load_rag_system_v11()

rag_chain = st.session_state.rag_chain

# 6. 處理輸入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在檢索並整理法規..."):
                try:
                    result = rag_chain.invoke(prompt)
                    response_text = result["response"]
                    source_docs = result["context"]

                    st.write(response_text)

                    # 【優化 3】改善使用者體驗 (UX)
                    with st.expander("📚 查看最佳參考來源 (Top 4 Sources)", expanded=False):
                        if not source_docs:
                            st.info("沒有找到相關的來源文件。")
                        else:
                            # [關鍵策略]：雖然 AI 讀了 10 筆，但我們只顯示前 4 筆給使用者看
                            # 這樣可以過濾掉後面排名較低、較不相關的雜訊 (如第 12 頁)
                            top_k_display = 4

                            for i, doc in enumerate(source_docs[:top_k_display]):
                                try:
                                    page_num = int(doc.metadata.get('page', 0)) + 1
                                except:
                                    page_num = doc.metadata.get('page', 'Unknown')

                                source = os.path.basename(doc.metadata.get('source', 'Unknown'))

                                # 整理內文：去除多餘換行，讓閱讀更流暢
                                clean_content = doc.page_content.replace('\n', ' ')

                                # 使用醒目的標題格式
                                st.markdown(f"### 🏅 來源 {i + 1}: 第 {page_num} 頁")

                                # 顯示完整內文 (不截斷)，使用引用區塊格式
                                st.info(clean_content)

                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")
    else:
        st.error("系統初始化失敗，無法執行回答。")