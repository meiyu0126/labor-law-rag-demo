import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# 1. 載入環境變數 (API Key)
load_dotenv()

# 設定路徑
FILE_PATH = os.path.join("data", "labor_law.pdf")
CHROMA_PATH = "chroma_db"  # 向量資料庫要存放在哪個資料夾


def create_vector_db():
    # --- 步驟 A: 讀取與切分 (跟剛剛一樣) ---
    if not os.path.exists(FILE_PATH):
        print("❌ 找不到 PDF 檔案")
        return

    print("🚀 開始建立向量資料庫...")
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，"]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"📄 已切分出 {len(chunks)} 個片段。")

    # --- 步驟 B: 清理舊資料庫 (為了開發方便) ---
    # 如果資料庫資料夾已經存在，先刪除，避免重複塞入資料
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("🧹 已清除舊的資料庫內容。")

    # --- 步驟 C: Embedding 與 儲存 ---
    print("🧠 正在進行 Embedding (將文字轉為向量)...這需要一點時間...")

    # 使用 OpenAI 的 Embedding 模型 (text-embedding-3-small 是目前 CP 值最高的)
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    # 建立並儲存到 ChromaDB
    # 這一步會同時做兩件事：1.呼叫OpenAI API轉向量 2.存入本地資料夾
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )

    # 這裡不需要 db.persist()，新版 LangChain 會自動儲存
    print(f"✅ 成功！向量資料庫已建立於 '{CHROMA_PATH}' 資料夾中。")
    print(f"📊 資料庫內共有 {db._collection.count()} 筆資料。")


if __name__ == "__main__":
    create_vector_db()