# run_mcp_agent_v2.py
import asyncio
import json
import httpx
import threading
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b-instruct"

# 定義本地 MCP Server 的啟動參數 (維持不變，執行 mcp_server.py)
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"]
)

# 建立線程安全鎖，確保同一時間只有一個 MCP LLM 分析任務在後台執行
_mcp_lock = threading.Lock()

async def _mcp_analysis_worker(alert_info: dict):
    """
    真正的 MCP 異步通訊核心。
    透過 MCP 協定調用 Tool 獲取數據，並驅動本地大模型生成報告。
    """
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚙️  [MCP Agent] 正在建立標準 Stdio 管道連線...")
    
    try:
        async with stdio_client(server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                # 1. 初始化 MCP 協議
                await session.initialize()
                
                # 2. 自動列出並格式化 Server 端的 Tools
                tools_response = await session.list_tools()
                available_tools = []
                for t in tools_response.tools:
                    available_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.inputSchema
                        }
                    })
                
                # 3. 構造強烈語氣的工安指令 Prompt
                messages = [{
                    "role": "user", 
                    "content": f"【緊急工安指令】後台狀態機已觸發 LEVEL_3 告警！當前觸發煙囪為 {alert_info['chimney_id']}，"
                               f"時間為 {alert_info['timestamp']}。請你【必須】立刻調用 `get_latest_incident_log` 工具獲取最新煙囪日誌，"
                               f"不允許自行猜測，拿到數據後再生成一份專業的事故歸因簡報！"
                }]
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 🧠 [LLM 決策中] 正在評估工安 Context 並準備調用 MCP 工具...")
                
                # 4. 呼叫 Ollama 進行 Tool Calling 決策 (Temperature=0.0 強制不發瘋)
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        OLLAMA_URL,
                        json={
                            "model": MODEL_NAME,
                            "messages": messages,
                            "tools": available_tools,
                            "options": {"temperature": 0.0},
                            "stream": False
                        },
                        timeout=30.0
                    )
                    
                    res_json = response.json()
                    message = res_json.get("message", {})
                    
                    # 5. 處理工具調用
                    if "tool_calls" in message and message["tool_calls"]:
                        for tool_call in message["tool_calls"]:
                            tool_name = tool_call["function"]["name"]
                            print(f"\n🔥 [LLM 決策成功] 確定觸發 MCP 工具調用：`{tool_name}`")
                            
                            if tool_name == "get_latest_incident_log":
                                # 真正經由 MCP RPC 執行 Server 端函數
                                server_result = await session.call_tool(tool_name, arguments={})
                                print(f"📥 [MCP Server 回傳結構化數據] -> {server_result.content[0].text}")
                                
                                # 將工具回傳數據餵回 Context
                                messages.append(message)
                                messages.append({
                                    "role": "tool",
                                    "content": server_result.content[0].text,
                                    "name": tool_name
                                })
                                
                                # 6. 最終將簡報即時寫入本地 report.md
                                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📝 [LLM 報告生成中] 基於 MCP 數據，報告已即時同步至 report.md ...")
                                with open("report.md", "w", encoding="utf-8") as f:
                                    f.write(f"# 🚨 石化園區自動工安事件歸因報告\n\n> 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                                    async with client.stream("POST", OLLAMA_URL, json={
                                        "model": MODEL_NAME,
                                        "messages": messages,
                                        "stream": True
                                    }) as r:
                                        async for line in r.aiter_lines():
                                            if line:
                                                chunk = json.loads(line)
                                                content = chunk.get("message", {}).get("content", "")
                                                print(content, end="", flush=True) # 終端機照樣打字
                                                f.write(content) # 同步寫入檔案
                                                f.flush()
                    else:
                        print("\n⚠️ [系統提示] 大模型這一次沒有調用 MCP 工具，它直接回覆了以下內容：")
                        print(message.get("content", "無回覆內容"))
                        
    except Exception as e:
        print(f"\n❌ [MCP Agent 錯誤] 執行失敗: {e}")

def _thread_entry(alert_info: dict):
    """新線程的入口點，負責初始化該線程的 Asyncio 事件迴圈"""
    # 嘗試獲取鎖，如果此時有上一個告警還在生成報告，則直接跳過，防止邊緣端 VRAM 爆炸
    acquired = _mcp_lock.acquire(blocking=False)
    if not acquired:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🛡️  [MCP Agent 互斥] 後台已有 LLM 任務正在推論，跳過本次重複請求。")
        return
    
    try:
        asyncio.run(_mcp_analysis_worker(alert_info))
    finally:
        _mcp_lock.release()

def trigger_mcp_analysis_async(alert_info: dict):
    """
    🚀 供 OpenCV 主程式呼叫的外部接口。
    採用完全獨立的守護線程（Daemon Thread）啟動，絕對零阻塞主執行緒的 OpenCV 畫面刷新與 FPS。
    """
    thread = threading.Thread(
        target=_thread_entry,
        args=(alert_info,),
        daemon=True
    )
    thread.start()

if __name__ == "__main__":
    # 單獨執行此檔案時的防呆測試
    mock_alert = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "chimney_id": "Chimney-Test-A",
    }
    print("=== 啟動 MCP Agent 獨立測試 ===")
    trigger_mcp_analysis_async(mock_alert)
    # 保持主線程存活以供測試觀察
    time.sleep(15)