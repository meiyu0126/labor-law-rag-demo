#Streamlit是目前Python界最紅的快速架站工具
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
#套件名稱,架構角色,功能說明 (Why do we need it?)
#langchain,總指揮 (Orchestrator),這是核心框架。它負責把 LLM、資料庫、文件讀取器串接起來。就像 Java 的 Spring Framework，負責管理整個應用程式的流程。
#langchain-community,擴充模組庫 (Extensions),LangChain 在最近的版本改版了，將第三方整合 (Integrations) 拆分出來。要使用大多數的工具 (如文件載入器、工具箱) 都需要它。
#langchain-openai,大腦介面 (Model Interface),專門用來跟 OpenAI API (GPT-3.5/4o) 對接的驅動程式。
#chromadb,向量資料庫 (Vector Store),這是 RAG 的長期記憶。它將文字轉換成向量 (Embeddings) 並儲存在本地端，讓我們可以用「語意」來搜尋資料，而不僅僅是關鍵字比對。
#pypdf,資料讀取器 (Parser),我們的 ETL 工具。用來從 PDF 檔案中提取純文字，讓程式能夠「讀懂」勞基法文件。
#tiktoken,計量單位 (Tokenizer),這是 OpenAI 開發的 Token 計算器。我們用它來計算字數與成本，並確保送給 AI 的文字量不會超過它的 Context Window 上限。
#python-dotenv,金鑰管理 (Config Manager),用來讀取 .env 檔案中的設定。這是資安最佳實踐，避免把 API Key 硬寫在程式碼裡 (Hard-code)。
# 1. 設定頁面
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
    st.info(f"Top-K: 3 (Strict)") # 顯示目前的設定

    if uploaded_file:
        st.success(f"目前使用文件：\n{uploaded_file.name}")
    else:
        st.warning("目前使用預設文件：\n勞動基準法.pdf")
# -------------------------

# 2. 建立資料庫
def build_vector_db_in_memory(file_path, embedding_function):
    try:
        file_name = os.path.basename(file_path)
        print(f"--- 開始處理檔案: {file_name} ---")

        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("❌ 錯誤: PDF 內容為空")
            return None

        # 切分設定
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", "！", "？", "，"]
        )
        chunks = text_splitter.split_documents(docs)

        # 過濾雜訊
        clean_chunks = [c for c in chunks if len(c.page_content) > 50]
        # 1. 篩選出長度 <= 50 的片段 (原本被丟棄的部分)
        noise_chunks = [c for c in chunks if len(c.page_content) <= 50]

        print(f"🔍 共發現 {len(noise_chunks)} 筆被過濾的內容。\n")
        print("以下列出前 5 筆範例供檢查：")
        print("=" * 40)

        # 2. 列印出來檢查 (為了避免洗版，這裡只先印前 5 筆)
        for i, c in enumerate(noise_chunks[:5]):
            content = c.page_content.strip()  # 去除前後空白讓顯示更整齊
            length = len(c.page_content)

            print(f"【被過濾片段 #{i + 1}】 (長度: {length})")
            print(f"內容: {content}")
            print("-" * 20)

        print(f"📄 切分完成，共 {len(clean_chunks)} 筆有效片段")

        # 產生唯一 ID
        import re
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', file_name)[:30]
        unique_id = int(time.time())
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


# 3. 載入系統
@st.cache_resource(show_spinner=False)
def load_rag_system(target_file_path):
    load_dotenv()

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

    db = build_vector_db_in_memory(target_file_path, embedding_function)
    if db is None: return None

    # 【關鍵修改】
    # 1. k=3: 只取前 3 名，砍掉第 4 名以後的雜訊。
    # 2. lambda_mult=0.7: 稍微調高相似度權重，減少因為「追求多樣」而抓到退休金的情況。
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 20,
            "lambda_mult": 0.8
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



# 4. 處理檔案邏輯
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
else:
    tmp_file_path = os.path.join("data", "labor_law.pdf")

# 5. 初始化 Session
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好！請上傳 PDF 文件，或直接詢問勞基法相關問題。"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 6. 載入系統
if "rag_chain" not in st.session_state or st.session_state.get("current_file") != tmp_file_path:
    with st.spinner("🚀 正在優化檢索模型..."):
        chain = load_rag_system(tmp_file_path)
        st.session_state.rag_chain = chain
        st.session_state.current_file = tmp_file_path
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
                        #st.expander:使用 Streamlit 的摺疊元件來收納來源資料
                        #expanded=True:設定 st.expander的「預設狀態」為展開
                        with st.expander("📚 查看最佳參考來源 (Top 3)", expanded=True):
                            for i, doc in enumerate(source_docs):
                                try:
                                    #取得頁碼邏輯
                                    page_idx = doc.metadata.get('page', 0)
                                    page_num = int(page_idx) + 1
                                except:
                                    page_num = "?"

                                source = os.path.basename(doc.metadata.get('source', 'Unknown'))
                                #去除PDF切分時產生的多餘換行符號，讓文字在UI上的閱讀體驗更流暢。
                                content = doc.page_content.replace('\n', ' ')

                                st.markdown(f"### 🏅 來源 {i + 1}: 第 {page_num} 頁")
                                st.info(content)
                    #將response_text存入st.session_state.messages列表;為了讓這則回答成為下一次呼叫 format_chat_history 時的一部分，形成完整的對話上下文 (Context Loop)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"發生錯誤：{e}")
    else:
        st.error("系統初始化失敗。")