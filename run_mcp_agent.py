# run_mcp_agent.py
import asyncio
import json
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b-instruct"

# 定義本地 MCP Server 的啟動參數 (執行你寫的 mcp_server.py)
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"]
)

async def main():
    print("[MCP Client] 正在建立與 MCP 煙囪監控伺服器的標準 Stdio 連線...")
    
    # 透過 stdio 建立管道
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. 初始化 MCP 協議
            await session.initialize()
            
            # 2. 列出 Server 目前支援的所有 Tools
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
            
            print(f"[MCP Client] 成功對接 Tool: {available_tools[0]['function']['name']}")
            
            # 3. 模擬工安狀態機觸發告警（加強語氣：強制要求調用工具）
            messages = [{
                "role": "user", 
                "content": "【緊急工安指令】後台狀態機已觸發 LEVEL_3 告警！請你【必須】立刻調用 `get_latest_incident_log` 工具獲取最新煙囪日誌，不允許自行猜測，拿到數據後再生成歸因簡報！"
            }]
            
            print("\n[LLM] 思考中（判斷是否需要調用 MCP 工具）...")
            
            # 呼叫 Ollama，並把 MCP Server 的工具格式餵給它
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    OLLAMA_URL,
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "tools": available_tools, # 傳遞 MCP 工具
                        "options": {"temperature": 0.0}, # 將溫度調為 0，讓它的決策極度確定、不隨機發瘋
                        "stream": False
                    },
                    timeout=30.0
                )
                
                res_json = response.json()
                message = res_json.get("message", {})
                
                # 4. 判斷大模型是否決定調用 MCP 工具
                if "tool_calls" in message and message["tool_calls"]:
                    for tool_call in message["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        print(f"\n🔥 [LLM 決策] 觸發 MCP 工具調用：`{tool_name}`")
                        
                        if tool_name == "get_latest_incident_log":
                            # 真正透過 MCP 協定去執行 Server 端的 Function
                            server_result = await session.call_tool(tool_name, arguments={})
                            print(f"📥 [MCP Server 回傳結構化數據] -> {server_result.content[0].text}")
                            
                            # 將工具回傳結果塞回對話歷史
                            messages.append(message)
                            messages.append({
                                "role": "tool",
                                "content": server_result.content[0].text,
                                "name": tool_name
                            })
                            
                            # 5. 讓大模型根據 MCP 數據生成最終的硬核工安報告
                            print("\n[LLM] 正在根據 MCP 數據 Stream 生成最終歸因簡報：\n")
                            async with client.stream("POST", OLLAMA_URL, json={
                                "model": MODEL_NAME,
                                "messages": messages,
                                "stream": True
                            }) as r:
                                async for line in r.aiter_lines(): # 這裡已修正為 aiter_lines()
                                    if line:
                                        chunk = json.loads(line)
                                        print(chunk.get("message", {}).get("content", ""), end="", flush=True)
                else:
                    # 防呆機制：如果模型偷懶不調用 Tool，我們把它回覆的純文字印出來看看
                    print("\n⚠️ [系統提示] 大模型這一次沒有調用 MCP 工具，它直接回覆了以下內容：")
                    print(message.get("content", "無回覆內容"))

if __name__ == "__main__":
    asyncio.run(main())