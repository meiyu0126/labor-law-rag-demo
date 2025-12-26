import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

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
    # 根據你的截圖，法規內容都在 class="law-reg-content" 裡面
    law_content = soup.find(class_="law-reg-content")

    if law_content:
        # 【關鍵步驟 2】找出每一條法規
        # 截圖顯示每一條都是一個 class="row" 的 div
        # 我們找出所有在 law_content 裡面的 row
        all_rows = law_content.find_all(class_="row")
        print(f"🔍 共發現 {len(all_rows)} 個段落 (包含條文與章節標題)...\n")

        for row in all_rows:
            # 【關鍵步驟 3】分離條號與內文
            # 左邊：class="col-no" (條號)
            col_no = row.find(class_="col-no")
            # 右邊：class="col-data" (內文)
            col_data = row.find(class_="col-data")

            # 只有當「條號」和「內文」同時存在時，才算是一條完整的法規
            # (因為有時候 row 裡面放的是 "第 一 章 總則" 這種章節標題，它沒有 col-data)
            if col_no and col_data:
                article_no = col_no.get_text(strip=True)  # 取得 "第 1 條"
                article_text = col_data.get_text(strip=True)  # 取得內文

                # 印出來檢查看看
                print(f"📌 {article_no}")
                print(f"📝 {article_text[:50]}...")  # 只印前50字
                print("-" * 20)

                # 【關鍵步驟 4】封裝成 Document
                # 這裡我們做一個很棒的優化：把條號直接寫進 Metadata！
                new_doc = Document(
                    page_content=f"{article_no}：{article_text}",  # 內容格式：第 1 條：內文...
                    metadata={
                        "source": "勞動基準法",
                        "url": url,
                        "article_id": article_no  # 這樣以後可以精準搜尋 "第 24 條"
                    }
                )
                crawled_docs.append(new_doc)

        print(f"\n📦 成功轉換 {len(crawled_docs)} 條法規為 LangChain 文件物件！")

    else:
        print("❌ 找不到 class='law-reg-content'，可能是網頁改版了？")

else:
    print("❌ 網頁讀取失敗")