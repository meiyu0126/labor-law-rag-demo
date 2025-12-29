import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tempfile
import time
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

load_dotenv()
#從全國法規資料庫抓取勞動基準法
# 加上這行，Streamlit 會把爬下來的結果存起來，不會每次都重跑
@st.cache_data(ttl=3600) # ttl=3600 代表快取 1 小時後過期
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

        #鎖定「大寶箱」
        law_content = soup.find(class_="law-reg-content")

        if law_content:
            #找出每一條法規
            all_rows = law_content.find_all(class_="row")
            print(f"🔍 共發現 {len(all_rows)} 個段落 (包含條文與章節標題)...\n")

            for row in all_rows:
                #分離條號與內文
                col_no = row.find(class_="col-no")
                col_data = row.find(class_="col-data")
                #BeautifulSoup 最常用的方法 .get_text();它會把 HTML 標籤（<div>...</div>）丟掉，只留下裡面的字。
                #strip=True;加了 strip=True：它會自動把前後的換行符號 (\n) 和多餘空白切除，變成乾淨的
                if col_no and col_data:
                    article_no = col_no.get_text(strip=True)
                    article_text = col_data.get_text(strip=True)

                    #封裝成 Document
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

            return crawled_docs

        else:
            print("❌ 找不到 class='law-reg-content'，可能是網頁改版了？")
            return []  # 失敗時回傳空列表

    else:
        print("❌ 網頁讀取失敗")
        return []

#套件名稱,架構角色,功能說明 (Why do we need it?)
#langchain,總指揮 (Orchestrator),這是核心框架。它負責把 LLM、資料庫、文件讀取器串接起來。就像 Java 的 Spring Framework，負責管理整個應用程式的流程。
#langchain-community,擴充模組庫 (Extensions),LangChain 在最近的版本改版了，將第三方整合 (Integrations) 拆分出來。要使用大多數的工具 (如文件載入器、工具箱) 都需要它。
#langchain-openai,大腦介面 (Model Interface),專門用來跟 OpenAI API (GPT-3.5/4o) 對接的驅動程式。
#chromadb,向量資料庫 (Vector Store),這是 RAG 的長期記憶。它將文字轉換成向量 (Embeddings) 並儲存在本地端，讓我們可以用「語意」來搜尋資料，而不僅僅是關鍵字比對。
#pypdf,資料讀取器 (Parser),我們的 ETL 工具。用來從 PDF 檔案中提取純文字，讓程式能夠「讀懂」勞基法文件。
#tiktoken,計量單位 (Tokenizer),這是 OpenAI 開發的 Token 計算器。我們用它來計算字數與成本，並確保送給 AI 的文字量不會超過它的 Context Window 上限。
#python-dotenv,金鑰管理 (Config Manager),用來讀取 .env 檔案中的設定。這是資安最佳實踐，避免把 API Key 硬寫在程式碼裡 (Hard-code)。
# 2. 設定頁面
st.set_page_config(page_title="企業智能問答助手", page_icon="📂")
st.title("📂 企業智能文件問答助手")
st.caption("🚀 Powered by Large Model")

# --- 側邊欄 ---
with st.sidebar:
    st.header("📂 文件上傳")
    uploaded_file = st.file_uploader("請上傳您的 PDF 文件", type=["pdf"])

    st.divider()
    st.header("⚙️ 系統參數")
    st.info(f"Chunk Size: 600")
    st.info(f"Chunk Overlap: 30")
    st.info(f"Top-K: 2(Strict)") # 顯示目前的設定

    if uploaded_file:
        st.success(f"目前使用文件：\n{uploaded_file.name}")
    else:
        st.warning("目前使用預設文件：\n勞動基準法")
# -------------------------

# 3. 建立資料庫(支援 PDF 路徑 或 Document 列表)
def build_vector_db_in_memory(source_data, embedding_function, is_web_data=False,original_filename=None):
    """
        source_data: 可以是檔案路徑 (str) 或是文件列表 (list)
        is_web_data: 標記是否為網路爬蟲資料;is_web_data=true->網路爬蟲資料
    """
    try:
        # --- 分支 A: 處理 PDF 檔案 ---
        if not is_web_data:
            file_path = source_data
            #如果有傳入原始檔名，就用原始檔名；否則用路徑檔名
            file_name = original_filename if original_filename else os.path.basename(file_path)

            print(f"--- 開始處理 PDF 檔案: {file_name} ---")

            loader = PyPDFLoader(file_path)
            docs = loader.load()
            if not docs:
                print("❌ 錯誤: PDF 內容為空")
                return None

            # 強制把 Metadata 裡的 source 改回原始檔名
            # 這樣 UI 顯示時，才會是 "88十個童女.pdf" 而不是 "tmpxyz.pdf"
            if original_filename:
                for doc in docs:
                    doc.metadata['source'] = original_filename

            # PDF 需要切分 (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=30,
                separators=["\n\n", "\n", "。", "！", "？", "，"]
            )
            chunks = text_splitter.split_documents(docs)
        # --- 分支 B: 處理網路爬蟲資料 ---
        else:
            print("--- 開始處理網路爬蟲資料 ---")
            docs = source_data  # source_data 是 List[Document]
            if not docs:
                print("❌ 錯誤: 爬蟲資料為空")
                return None
            # 網路爬蟲的資料每一條就是一條法規，通常不需要再切分，或者簡單切分即可
            # 這裡我們直接把它當作 chunks 使用 (因為每一條法規長度適中)
            chunks = docs
            file_name = "web_labor_law"  # 給個假檔名
        # --- 共同流程: 過濾雜訊與建庫 ---
        # 過濾太短的雜訊
        clean_chunks = [c for c in chunks if len(c.page_content) > 10]
        print(f"📄 有效片段共 {len(clean_chunks)} 筆")

        #以下這段程式碼的目的是為了給向量資料庫（ChromaDB）產生一個 「合法、安全且絕對唯一」 的 Collection 名稱。
        # 因為資料庫對於名稱的規定通常很嚴格（例如：不能有空格、不能有特殊符號、不能太長），且我們不希望新的檔案覆蓋掉舊的檔案，所以需要這段「整形手術」。
        import re
        # re.sub(正則表達式, 替換成什麼, 目標字串);Python 的正則表達式取代功能
        #r'[^a-zA-Z0-9]';[]：代表字元集合;^：代表「非」 (Not);a-zA-Z0-9：代表所有英文大小寫字母與數字
        #r'[^a-zA-Z0-9]':只要不是英文字母或數字的字元（包含中文、空格、點、括號），全部都抓出來
        #'_'：把抓出來的那些「非法字元」，全部替換成底線 _。
        #[:30]：字串切片。不管檔名多長，只取前 30 個字。這是為了避免超過 ChromaDB 的名稱長度限制（通常限制 63 字元）。
        #目的：確保檔名只剩下 ASCII 安全字元，不會讓資料庫報錯。範例： 如果 file_name 是 "勞基法 V1.0.pdf",結果： safe_name 會變成 "____V1_0_pdf" (中文和點都被變底線了)。
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', file_name)[:30]
        #time.time()：取得目前的 Unix 時間戳記（從 1970/1/1 到現在經過的秒數），例如 1735118855.123。
        #int(...)：轉成整數，去掉小數點。
        #目的： 確保 「唯一性 (Uniqueness)」。
        # 即使上傳同一個檔案 labor_law.pdf 兩次，因為時間不同，產生的 ID 就會不同，系統就不會搞混或錯誤覆蓋。
        unique_id = int(time.time())
        #組裝最終名稱 (Assembly)
        #f"..."：Python 的 f-string 格式化字串。
        #結構： 固定前綴 (rag_) + 清洗後的檔名 + 時間戳記。
        #目的： 產生一個人類稍微看得懂（知道是 RAG 用的，也大概知道是哪個檔），且機器絕對讀得懂的 ID。
        #例如:rag_______2025__pdf_1735000000
        #ChromaDB 對 Collection Name 有嚴格的命名規範（通常要求由字母數字或底線組成，且長度有限制）。 此外，為了支援多版本管理或避免同名檔案衝突，加上time.time() 時間戳記，確保每次上傳建立的資料庫都是獨立且唯一的實體，這增加了系統的穩健性。
        collection_name = f"rag_{safe_name}_{unique_id}"

        db = Chroma.from_documents(
            documents=clean_chunks,
            embedding=embedding_function,
            collection_name=collection_name
        )
        print(f"✅ 資料庫建立成功 (ID: {unique_id})！")
        return db

    except Exception as e:
            print(f"❌ 建立失敗: {e}")
            return None

# 4. 載入系統
@st.cache_resource(show_spinner=False)
# st.cache_resource 會自動檢查輸入參數 target_source (即爬蟲抓下來的 Document 列表) 的內容雜湊值 (Hash) 是否改變。
# 若 fetch_labor_law_docs 的 1 小時快取過期 (Expire)，當使用者送出查詢 (Submit) 時，
# 系統會強制重新爬取最新法條。若法條內容有更新，target_source 就會改變，進而觸發這裡重新建立向量資料庫。
def load_rag_system(target_source,is_web=False,original_filename=None):

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")
    # 呼叫修改後的建庫函式
    db = build_vector_db_in_memory(target_source, embedding_function, is_web_data=is_web,original_filename=original_filename)
    if db is None: return None

    # 1. k=2: 只取前2名
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 20,
            "lambda_mult": 0.80
        }
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    template = """你是一個專業的文件問答助手。
    請依據【參考資料】與【歷史對話】來回答使用者的問題。

    【歷史對話】：
    {chat_history}

    【參考資料】：
    {context}

    使用者問題：{question}

    回答："""

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    from operator import itemgetter

    retrieval_step = RunnableParallel(
        {
            "context": itemgetter("question") | retriever ,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    )

    answer_step = (
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
    )
    # 設定最終輸出的平行處理 (Parallel Execution)：
    # 1. "response": 負責生成回答
    #    - 接收檢索結果 -> 格式化為字串 (String) -> 組裝 Prompt -> LLM 推論輸出
    # 2. "context": 負責保留原始證據
    #    - 直接保留檢索到的原始文件物件 (List[Document])，用於前端顯示來源(使前端能拿到 metadata（頁碼、檔名）)
    final_chain = retrieval_step | RunnableParallel({
        "response": answer_step,
        "context": lambda x: x["context"]
    })

    return final_chain


# --- 歷史訊息處理 ---
def format_chat_history(messages):
    history_text = ""
    recent_messages = messages[-6:]
    for msg in recent_messages:
        if msg["role"] == "user":
            history_text += f"使用者: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"助手: {msg['content']}\n"
    return history_text

# 5. 初始化 Session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！請上傳 PDF 文件，或直接詢問勞基法相關問題。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 決定資料來源
target_source = None
is_web = False
current_file_id = "default_web" # 用來識別檔案是否有變更
real_name = None #初始化變數

if uploaded_file:
    # 如果有上傳檔案，走 PDF 流程
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        target_source = tmp_file.name
        is_web = False
        current_file_id = uploaded_file.name
        #抓取使用者上傳的原始檔名
        real_name = uploaded_file.name
else:
    # 如果沒上傳，走網路爬蟲流程
    target_source = fetch_labor_law_docs()
    is_web = True
    current_file_id = "web_labor_law"

# 6. 載入系統
# 判斷是否需要重新建立 (檔案變了 OR 系統還沒初始化)
if "rag_chain" not in st.session_state or st.session_state.get("current_file") != current_file_id:
    with st.spinner("🚀 正在建置知識庫 (PDF/Web)..."):
        # 傳入 source 和 標記
        chain = load_rag_system(target_source, is_web=is_web, original_filename=real_name)

        st.session_state.rag_chain = chain
        st.session_state.current_file = current_file_id

rag_chain = st.session_state.rag_chain

# 7. 處理輸入
#賦值表達式;python3.8之後的功能,st.chat_input()將值賦予prompt,由if做條件判斷,判斷prompt是否有值,若有則不為None
if prompt := st.chat_input():
    #當使用者輸入並送出時，程式會做兩件事:
    #將訊息存入 st.session_state.messages (List)，這是為了讓 AI 有短期記憶
    st.session_state.messages.append({"role": "user", "content": prompt})
    #使用 st.chat_message("user").write(prompt) 立即將訊息顯示在聊天視窗上
    st.chat_message("user").write(prompt)

    if rag_chain:
        #負責建立UI容器,會在畫面上畫出一個「對話框區域」，並在左側（預設）顯示一個機器人的頭像,確保後續的輸出都歸類在『助手』的對話框內
        #就像是漫畫裡的一個「對話泡泡」，並標示這是「助手」說的話
        #生命週期： 永久存在（直到被捲動或是重新整理）。
        with st.chat_message("assistant"):
            #負責提供即時回饋;
            #RAG的檢索與生成需要時間，利用這個暫時性的 Spinner 告訴使用者系統正在運作，避免使用者以為網頁當機。
            #當運算結束離開內層 with 區塊時，Spinner 會自動消失，無縫切換顯示最終的回答文字
            #生命週期： 暫時的(Temporary)
            with st.spinner("🔍 正在檢索..."):
                try:
                    #在呼叫 AI 前，先整理過去的對話紀錄
                    # messages[:-1]代表不包含最新的這句，避免重複。
                    history_str = format_chat_history(st.session_state.messages[:-1])
                    #result是rag_chain呼叫.invoke()後執行的結果,以包含question,chat_history的字典為參數
                    #rag_chain是執行load_rag_system後回傳的Chain實體,其輸出結構 (Output Schema)包括:1.response 2.context 這2個key
                    #response的產生過程(LCEL(LangChain Expression Language)的管線化)：資料(question,chat_history)先流經 retrieval_step 取得資訊，再傳遞給 answer_step 進行格式化(context轉為字串),產生prompt傳給LLM生成回覆
                    #context則是經過檢索之後得到的原始資料
                    #result是rag_chain執行.invoke()後的產出，結構對應final_chain的定義包括了response,context
                    #gemini提供的註解如下
                    # [Input]: 準備參數
                    # 將 "當前問題" 與 "歷史紀錄" 打包成 Dictionary，作為 invoke 的輸入
                    # [Process]: 執行 RAG 鏈
                    # rag_chain 是由 load_rag_system 建構完成的物件 (即 final_chain)
                    # [Output]: 解析結果
                    # result 的結構由 final_chain 中的 RunnableParallel 定義：
                    # 1. result["response"]: 經過 retrieval_step (檢索) -> answer_step (生成) 後的 AI 回覆字串
                    # 2. result["context"]: 經過 retrieval_step 檢索到的原始 Document 物件列表 (原始資料)
                    result = rag_chain.invoke({
                        "question": prompt,
                        "chat_history": history_str
                    })

                    response_text = result["response"]
                    source_docs = result["context"]
                    st.write(response_text)

                    if source_docs:
                        with st.expander("📚 查看最佳參考來源 (Top 2)", expanded=True):
                            for i, doc in enumerate(source_docs):
                                # --- 智慧判斷來源類型 ---
                                # 如果有 'article_id' 代表是法規條文
                                if 'article_id' in doc.metadata:
                                    source_label = doc.metadata['article_id']  # 顯示 "第 24 條"
                                    #page_info = ""  # 法規不需要頁碼
                                # 否則就是 PDF，顯示頁碼
                                else:
                                    page_idx = doc.metadata.get('page', 0)
                                    source_label = f"第 {int(page_idx) + 1} 頁"

                                source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                content = doc.page_content.replace('\n', ' ')

                                st.markdown(f"### 🏅 來源 {i + 1}: {source_name} {source_label}")
                                st.info(content)
                    #將response_text存入st.session_state.messages列表;為了讓這則回答成為下一次呼叫 format_chat_history 時的一部分，形成完整的對話上下文 (Context Loop)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")
    else:
        st.error("系統初始化失敗。")