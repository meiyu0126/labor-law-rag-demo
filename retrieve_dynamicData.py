import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 【修改 1】把路徑改成你剛剛新建立的資料夾名稱
CHROMA_PATH = "chroma_db_web_version"


def search_test():
    # 【修改 2】確認模型跟建庫時一樣用 large
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # 測試問題
    query = "加班費怎麼算？"

    print(f"🔎 正在從【Web 版資料庫】搜尋問題：'{query}' ...")
    print("-" * 30)

    results = db.similarity_search_with_score(query, k=3)

    if not results:
        print("❌ 找不到相關資料。")
        return

    for i, (doc, score) in enumerate(results):
        print(f"🏆 [第 {i + 1} 名] (Score: {score:.4f})")
        # 這裡現在會顯示我們爬蟲抓到的 "第 XX 條"
        print(f"條號: {doc.metadata.get('article_id')}")
        print(f"來源: {doc.metadata.get('source')}")
        print(f"內容: {doc.page_content[:100]}...")
        print("-" * 30)


if __name__ == "__main__":
    search_test()