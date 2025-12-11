import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. 載入環境變數
load_dotenv()

CHROMA_PATH = "chroma_db"


def format_docs(docs):
    """將檢索到的多個片段合併成一段文字"""
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag_system():
    print("🤖 初始化 RAG 系統中...")

    # 2. 準備檢索器 (Retriever)
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 設定檢索器：只抓最相關的前 5 筆 (為了提高抓到第 24 條的機率，我們把 k 提高到 5)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 3. 準備 LLM (大腦)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 4. 設計 Prompt (指令)
    # 這一步最關鍵！我們要告訴 AI：「你只能根據我給你的資料回答，不要瞎掰。」
    template = """你是一個專業的勞基法問答助手。
    請依據以下的【參考資料】來回答使用者的問題。
    如果資料中沒有答案，請直接說「抱歉，根據目前的資料庫，我無法回答這個問題」，不要試圖憑空捏造。

    【參考資料】：
    {context}

    使用者問題：{question}

    回答："""

    prompt = ChatPromptTemplate.from_template(template)

    # 5. 建立 RAG 鏈 (Chain)
    # 這是 LangChain 的 LCEL 語法 (LangChain Expression Language)
    # 流程：取得問題 -> 檢索資料 -> 整理資料 -> 填入 Prompt -> 丟給 LLM -> 解析字串
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    # 6. 開始互動
    question = "請問加班費的計算標準是如何規定的？(請引用條文)"
    print(f"📝 正在詢問問題：{question}")
    print("-" * 50)

    # 執行！
    result = rag_chain.invoke(question)

    print("💡 AI 回答結果：")
    print(result)
    print("-" * 50)


if __name__ == "__main__":
    run_rag_system()