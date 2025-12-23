import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#用來測試的程式
# 1. 載入環境變數
# override=True 確保如果有系統變數，會以 .env 為主
load_dotenv(override=True)


def test_environment():
    print(f"🐍 Python 版本: {sys.version.split()[0]}")

    # 2. 檢查 Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ 錯誤：未讀取到 OPENAI_API_KEY，請檢查 .env 檔案存放位置是否在專案根目錄。")
        return

    print(f"✅ 金鑰讀取成功 (前五碼): {api_key[:5]}...")

    # 3. 測試 LLM 連線
    try:
        print("🤖 正在呼叫 OpenAI API (這可能需要幾秒鐘)...")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        response = llm.invoke("請用一句充滿熱情的話，鼓勵一位正在轉職 AI 工程師的資深開發者。")

        print("\n" + "=" * 40)
        print("💬 模型回應：")
        print(response.content)
        print("=" * 40 + "\n")
        print("🎉 環境建置完美成功！可以開始開發了！")

    except Exception as e:
        print(f"❌ 呼叫失敗：{e}")
        print("請檢查網路連線或 API Key 是否有額度。")


if __name__ == "__main__":
    test_environment()