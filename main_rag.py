import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#將檢索到的這些片段（Context）餵給 GPT，看它能不能回答出來。
# 1. 載入環境變數
load_dotenv()

CHROMA_PATH = "chroma_db"


def format_docs(docs):
    """將檢索到的多個片段合併成一段文字"""
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag_system():
    print("🤖 初始化 RAG 系統中...")

    # 2. 準備檢索器 (Retriever)
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 設定檢索器：只抓最相關的前 5 筆 (為了提高抓到第 24 條的機率，我們把 k 提高到 5)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 3. 準備 LLM (大腦)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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
    #分支一 "question": RunnablePassthrough()：RunnablePassthrough()是一個「直通車」,把 question 原封不動地傳給 "question" 這個 Key
    #分支二 "context": retriever | format_docs：
    #輸入的 "加班費怎麼算？" 先傳給 retriever（檢索器），找出相關的 Document 物件列表
    #接著把這些 Documents 傳給 format_docs 函式（通常是用 \n\n 接起來），轉成一個長字串。
    #最後這個長字串被傳給 "context" 這個 Key。
    #所以rag_chain的產生:1.產生key為context,question的字典
    # 2.LangChain 會自動把字典裡的 context 和 question 填入 Prompt Template 對應的 {context} 與 {question} 預留位置中。
    #產出：一個完整的 PromptValue 物件,包括了System message("你是一個專業助手..."),User Message (有 "文件內容" + "問題")。
    #傳給LLM做推論,呼叫 OpenAI（或其他模型）進行預測,產出一個 AIMessage 物件（例如 AIMessage(content="加班費的計算方式是...")）。
    #StrOutputParser()接收：AIMessage 物件後把 content 裡面的文字內容萃取出來,產出單純的字串 (String)（這就是最後 invoke 回傳的東西）
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