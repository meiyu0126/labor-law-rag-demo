import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1. 設定頁面
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手")
st.caption("🚀 Powered by RAG (LangChain + ChromaDB + OpenAI)")


# 2. 載入資料庫
@st.cache_resource
def load_rag_system():
    load_dotenv()
    CHROMA_PATH = "chroma_db"

    if not os.path.exists(CHROMA_PATH):
        st.error("❌ 找不到向量資料庫，請確認已執行 ingest.py 並將 chroma_db 上傳至 GitHub！")
        return None

    # 使用與 ingest.py 相同的模型
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 設定檢索器 (k=5)
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

    # 定義格式化文件的函式
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # --- 修正後的 RAG 鏈結邏輯 (更穩定) ---

    # 步驟 1: 平行處理 - 一邊去抓資料(context)，一邊保留使用者問題(question)
    retrieval_step = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    # 步驟 2: 生成回答 - 將抓到的資料格式化成字串，然後餵給 LLM
    answer_step = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
    )

    # 步驟 3: 組合最終輸出 - 回傳「AI回答」以及「原始文件(用於顯示來源)」
    final_chain = retrieval_step | RunnableParallel({
        "response": answer_step,
        "context": lambda x: x["context"]
    })

    return final_chain


rag_chain = load_rag_system()

# 3. 初始化對話
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手。請輸入你想查詢的勞基法問題："}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 4. 處理輸入
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在檢索法規資料庫..."):
                try:
                    # 執行 RAG
                    result = rag_chain.invoke(prompt)

                    response_text = result["response"]
                    source_docs = result["context"]

                    # 顯示回答
                    st.write(response_text)

                    # 顯示資料來源 (Expander)
                    with st.expander("📚 查看資料來源 (Source Documents)"):
                        if not source_docs:
                            st.info("沒有找到相關的來源文件。")
                        else:
                            for i, doc in enumerate(source_docs):
                                page = doc.metadata.get('page', 'Unknown')
                                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                st.markdown(f"**來源 {i + 1}**: `{source}` (第 {page} 頁)")
                                st.text(doc.page_content[:100] + "...")  # 只顯示前100字預覽
                                st.divider()

                    # 更新紀錄
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")