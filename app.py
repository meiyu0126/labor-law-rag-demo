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
st.title("⚖️ 企業勞基法智慧問答助手 (V9 - In-Memory)")
st.caption("🚀 Powered by RAG (In-Memory DB + Cache)")


# 2. 定義建立資料庫函式 (純邏輯)
# 這次我們不存檔，直接回傳 DB 物件
def build_vector_db_in_memory(file_path, embedding_function):
    try:
        print(f"--- [V9] 開始建立記憶體資料庫 ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("❌ 錯誤: PDF 內容為空")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)
        print(f"📄 切分完成，共 {len(chunks)} 筆片段")

        # 【關鍵修改】：不指定 persist_directory，就會在記憶體中執行
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function
            # persist_directory=None <--- 不寫這行就是 In-Memory
        )
        print("✅ 記憶體資料庫建立成功！")
        return db
    except Exception as e:
        print(f"❌ 建立失敗: {e}")
        return None


# 3. 載入 RAG 系統 (使用快取)
# 因為 DB 現在在記憶體，我們必須用 cache_resource 把它留住，不然每次互動都會消失
@st.cache_resource(show_spinner=False)  # 關閉內建 spinner，我們自己要在外面畫
def load_rag_system_v9():
    load_dotenv()

    FILE_PATH = os.path.join("data", "labor_law.pdf")
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    # 直接建立 (不檢查資料夾了，因為沒有資料夾)
    db = build_vector_db_in_memory(FILE_PATH, embedding_function)

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


# 4. 初始化 Session & 載入系統
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手 (V9)。請輸入你想查詢的勞基法問題："}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 5. 呼叫載入 (加上外部轉圈圈)
if "rag_chain" not in st.session_state:
    with st.spinner("🚀 系統啟動中... 正在記憶體中構建知識庫 (約 20 秒)..."):
        # 這裡會觸發函式，如果已經快取過，會瞬間完成
        st.session_state.rag_chain = load_rag_system_v9()

rag_chain = st.session_state.rag_chain

# 6. 處理輸入
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
    else:
        st.error("系統初始化失敗，無法執行回答。")