import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# 1. 設定頁面標題
st.set_page_config(page_title="勞基法 AI 助手", page_icon="⚖️")
st.title("⚖️ 企業勞基法智慧問答助手")
st.caption("🚀 Powered by RAG (LangChain + ChromaDB + OpenAI)")


# 2. 載入環境與資料庫 (利用 cache resource 加速)
@st.cache_resource
def load_rag_system():
    load_dotenv()
    CHROMA_PATH = "chroma_db"

    # 檢查資料庫是否存在
    if not os.path.exists(CHROMA_PATH):
        st.error("❌ 找不到向量資料庫，請先執行 ingest.py 建立資料庫！")
        return None

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 修改後的 retriever 設定
    retriever = db.as_retriever(
        search_type="similarity_score_threshold",  # 1. 啟用「門檻過濾」模式
        search_kwargs={
            "k": 5,  # 最多還是抓 5 筆
            "score_threshold": 0.5  # 2. 設定門檻：相似度低於 0.7 的直接丟掉
        }
    )


    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    template = """你是一個專業的勞基法問答助手。
    請依據以下的【參考資料】來回答使用者的問題。
    如果資料中沒有答案，請直接說「抱歉，根據目前的資料庫，我無法回答這個問題」，不要試圖憑空捏造。

    【參考資料】：
    {context}

    使用者問題：{question}

    回答："""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # === 🔥 關鍵修改開始：使用 RunnableParallel 來保留來源文件 ===

    # 1. 先定義檢索步驟：同時取得「文件(context)」和「問題(question)」
    retrieval_step = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    # 2. 定義回答生成步驟：把 context 轉成字串 -> 丟給 Prompt -> LLM
    answer_step = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
    )

    # 3. 組合最終鏈：同時回傳「原始文件 (source_documents)」和「AI回答 (answer)」
    rag_chain = (
            retrieval_step
            | RunnableParallel({
        "source_documents": lambda x: x["context"],
        "answer": answer_step
    })
    )
    # === 🔥 關鍵修改結束 ===

    return rag_chain


# 初始化 RAG 鏈
rag_chain = load_rag_system()

# 3. 處理對話歷史 (Session State)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好！我是你的勞基法 AI 助手。請問關於加班費、休假或工時，有什麼想問的嗎？"}]

# 顯示歷史訊息
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    # 如果歷史訊息中有來源資訊，也顯示出來 (可選)
    if "sources" in msg:
        with st.expander("查看參考來源"):
            for source in msg["sources"]:
                st.markdown(f"- **{source['source']}** (Page {source['page']})")

# 4. 處理使用者輸入
if prompt := st.chat_input():
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 生成 AI 回應
    if rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("🔍 正在檢索法規資料庫..."):
                # 呼叫 invoke，現在 response 會是一個字典 (Dictionary)
                result = rag_chain.invoke(prompt)

                answer = result["answer"]
                source_docs = result["source_documents"]

                # 顯示回答
                st.write(answer)

                # === 🔥 新增：顯示資料來源 ===
                # 整理來源資訊，避免重複顯示相同的頁數
                unique_sources = []
                seen_sources = set()

                for doc in source_docs:
                    # 取得檔名 (去除路徑) 和頁數
                    source_name = os.path.basename(doc.metadata.get("source", "未知來源"))
                    page_num = doc.metadata.get("page", 0) + 1  # 程式從0開始，習慣上加1顯示

                    identifier = f"{source_name}-{page_num}"
                    if identifier not in seen_sources:
                        unique_sources.append({"source": source_name, "page": page_num})
                        seen_sources.add(identifier)

                # 使用折疊元件 (Expander) 顯示來源
                with st.expander("📚 查看資料來源 (Source Documents)"):
                    for item in unique_sources:
                        st.markdown(f"- 📄 **{item['source']}** : 第 {item['page']} 頁")
                    st.caption("註：頁數為 PDF 原始頁碼")

        # 存入歷史紀錄 (包含來源資訊，以便重新整理頁面時也能顯示)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": unique_sources
        })