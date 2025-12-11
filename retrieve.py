import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# 1. 載入環境變數
load_dotenv()

CHROMA_PATH = "chroma_db"


def search_test():
    # 準備 Embedding Function (必須跟建立資料庫時用的一模一樣！)
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. 連接現有的向量資料庫
    # 注意：這裡不用再餵 documents，只要指定 persist_directory
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # 3. 模擬使用者提問
    # 你可以隨意修改這個問題，例如："加班費怎麼算？"、"特休假幾天？"
    query = "加班費怎麼算？"

    print(f"🔎 正在搜尋問題：'{query}' ...")
    print("-" * 30)

    # 4. 執行相似度搜尋 (Similarity Search)
    # k=3 代表我們要找出「最相關的前 3 筆」資料
    results = db.similarity_search_with_score(query, k=3)

    # 5. 展示搜尋結果
    if not results:
        print("❌ 找不到相關資料。")
        return

    for i, (doc, score) in enumerate(results):
        print(f"🏆 [第 {i + 1} 名] (相似度距離 Score: {score:.4f})")
        print(f"來源頁數: {doc.metadata.get('page')}")
        print(f"內容預覽: {doc.page_content[:100]}...")  # 只印出前100字避免洗版
        print("-" * 30)

    print("✅ 檢索測試完成！如果內容與問題相關，代表 RAG 成功了一半。")


if __name__ == "__main__":
    search_test()