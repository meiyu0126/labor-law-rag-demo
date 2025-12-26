import shutil

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# 【修正 1】一開始就載入環境變數，這樣後面的 OpenAIEmbeddings 才讀得到 Key
load_dotenv()


def fetch_labor_law_docs():
    # 1. 設定目標網址 (全國法規資料庫 - 勞動基準法)
    url = "https://law.moj.gov.tw/LawClass/LawAll.aspx?PCode=N0030001"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    print(f"🚀 開始爬取頁面：{url} ...")
    response = requests.get(url, headers=headers)

    # 先檢查狀態碼，確認有沒有連線成功
    print(f"📡 連線狀態碼: {response.status_code}")

    crawled_docs = []

    if response.status_code == 200:
        print("✅ 連線成功！開始解析 HTML...")
        soup = BeautifulSoup(response.text, "html.parser")

        # 【關鍵步驟 1】鎖定「大寶箱」
        law_content = soup.find(class_="law-reg-content")

        if law_content:
            # 【關鍵步驟 2】找出每一條法規
            all_rows = law_content.find_all(class_="row")
            print(f"🔍 共發現 {len(all_rows)} 個段落 (包含條文與章節標題)...\n")

            for row in all_rows:
                # 【關鍵步驟 3】分離條號與內文
                col_no = row.find(class_="col-no")
                col_data = row.find(class_="col-data")
                #BeautifulSoup 最常用的方法 .get_text();它會把 HTML 標籤（<div>...</div>）丟掉，只留下裡面的字。
                #strip=True;加了 strip=True：它會自動把前後的換行符號 (\n) 和多餘空白切除，變成乾淨的
                if col_no and col_data:
                    article_no = col_no.get_text(strip=True)
                    article_text = col_data.get_text(strip=True)

                    # 【關鍵步驟 4】封裝成 Document
                    new_doc = Document(
                        page_content=f"{article_no}：{article_text}",
                        metadata={
                            "source": "勞動基準法",
                            "url": url,
                            "article_id": article_no
                        }
                    )
                    crawled_docs.append(new_doc)

            print(f"\n📦 成功轉換 {len(crawled_docs)} 條法規為 LangChain 文件物件！")

            # 【修正 2】非常重要！一定要把結果回傳出去，不然外面拿到的是 None
            return crawled_docs

        else:
            print("❌ 找不到 class='law-reg-content'，可能是網頁改版了？")
            return []  # 失敗時回傳空列表

    else:
        print("❌ 網頁讀取失敗")
        return []


# 因為上面已經 load_dotenv() 了，這裡就能安全建立物件
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
CHROMA_PATH = "chroma_db_web_version"
if __name__ == "__main__":
    print("🕸️ 開始爬取最新法規...")
    docs = fetch_labor_law_docs()  # 拿到 98 條法規

    # 防呆機制：確認有抓到資料才存
    if docs:
        # 【新增這段】檢查並清除舊資料庫
        if os.path.exists(CHROMA_PATH):
            print(f"🧹 偵測到舊資料庫，正在清理：{CHROMA_PATH} ...")
            shutil.rmtree(CHROMA_PATH)  # 強制刪除整個資料夾
            print("✨ 舊資料清理完成！")
        print(f"💾 開始寫入向量資料庫 (共 {len(docs)} 筆)...")

        # 直接存進 DB
        db = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            persist_directory="./chroma_db_web_version"
        )
        print("🎉 資料庫建立完成！資料夾：chroma_db_web_version")
    else:
        print("⚠️ 沒有抓到任何資料，略過建庫步驟。")