# mcp_server.py
import json
from mcp.server.fastmcp import FastMCP

# 初始化 FastMCP 服務
mcp = FastMCP("ChimneyMonitor")

@mcp.tool()
def get_latest_incident_log() -> str:
    """
    獲取石化園區煙囪最新的工安異常狀態機日誌。
    回傳已過濾、結構化的精華數據，包含煙囪編號、不正常排煙顏色及持續時間。
    """
    # 模擬從你的時序狀態機模組提取出的精華數據
    incident_data = {
        "timestamp": "2026-06-09 13:30:00",
        "factory_zone": "A3 石化裂解區",
        "chimney_id": "CH-04",
        "anomaly_type": "Black Smoke (密濃黑煙)",
        "duration_seconds": 45,
        "opacity_level": "85%",
        "status": "TRIGGERED_ALARM_LEVEL_3"
    }
    return json.dumps(incident_data, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run(transport='stdio')