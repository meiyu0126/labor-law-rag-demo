import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#這支 etl.py 負責 RAG 系統的資料前處理;用來測試「讀取」跟「切分」
# 首先，使用 PyPDFLoader 將非結構化的 PDF 載入為文件物件。
# 接著，採用 RecursiveCharacterTextSplitter 進行切分，設定 Chunk Size 為 500 並搭配 50 的 Overlap。 這樣的策略是為了適應 LLM 的 Context Window 限制，同時透過 Overlap 保持語意連貫性，最後保留頁碼 Metadata，以支援前端的引用來源顯示功能。
# 設定資料路徑
FILE_PATH = os.path.join("data", "labor_law.pdf")


def load_and_split_pdf():
    # 1. 檢查檔案是否存在
    if not os.path.exists(FILE_PATH):
        print(f"❌ 錯誤：找不到檔案 {FILE_PATH}。請確認你有建立 data 資料夾並放入 pdf。")
        return

    print(f"📂 開始讀取檔案：{FILE_PATH} ...")

    # 2. 載入器 (Loader)：負責將 PDF 轉為純文字物件 (Document Object)
    #Extract (萃取)：使用 LangChain 內建的 PyPDFLoader將 PDF 轉為文字物件
    loader = PyPDFLoader(FILE_PATH)
    docs = loader.load()
    print(f"✅ 讀取成功！原始文件共有 {len(docs)} 頁。\n")

    # 3. 切分器 (Splitter)：RAG 的靈魂
    # 為什麼要切？因為 LLM 的 Context Window 有限，且我們希望搜尋時能精準定位到「某個條款」而非整本書。
    # chunk_size=500: 每個區塊約 500 字 (這對於法規條文來說通常包含 1-2 條完整條文)
    # chunk_overlap=50: 前後區塊重疊 50 字，避免把一句話切斷在中間，保留上下文連貫性。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""]  # 優先在段落或句點處切分
    )

    print("✂️ 正在進行文字切分 (Chunking)...")
    chunks = text_splitter.split_documents(docs)

    print(f"🎉 切分完成！總共切出了 {len(chunks)} 個片段 (Chunks)。")
    print("=" * 40)

    # 4. 驗證結果：印出前 3 個片段來檢查品質
    #這裡要檢查「條文」有沒有被硬生生切斷？
    for i, chunk in enumerate(chunks[:3]):
        print(f"📄 [片段 {i + 1}] (長度: {len(chunk.page_content)})")
        print(chunk.page_content)
        print("-" * 20)
        print(f"來源頁數: {chunk.metadata.get('page')}")  # 這是 Citation (引用來源) 的基礎
        print("=" * 40)


if __name__ == "__main__":
    load_and_split_pdf()