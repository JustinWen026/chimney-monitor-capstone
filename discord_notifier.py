import requests
import threading
from datetime import datetime

# =====================================================================
# Discord Webhook 設定區 (完全解耦，方便隨時修改 Webhook 網址)
# =====================================================================
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE"

def _send_discord_worker(timestamp, duration):
    """
    實際在背景執行緒中運行的 Discord 發送工作
    """
    if DISCORD_WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠️ [Discord] 尚未設定 Webhook URL，跳過發送。")
        return

    # 專為成大資工專題與海報視覺美化設計的卡片排版
    payload = {
        "username": "石化園區 AI 偵測系統",
        "avatar_url": "https://i.imgur.com/8nS47H3.png", # 機器人預設頭像
        "embeds": [
            {
                "title": "🚨 系統安全核心告警：偵測到異常排煙外溢 🚨",
                "description": "系統偵測到廠區煙囪排煙持續時間已達臨界值，已自動歸因並觸發主動防護通報機制。",
                "color": 15158332, # 鮮豔紅色邊條
                "fields": [
                    {"name": "📌 專案名稱", "value": "AI在石化園區工業煙囪煙霧安全偵測與其在事故歸因分析之研究", "inline": False},
                    {"name": "🔥 異常事件", "value": "煙霧外溢 (Smoky Event Detected)", "inline": True},
                    {"name": "⏱ 累積秒數", "value": f"`{duration} 秒` (門檻: 3.0秒)", "inline": True},
                    {"name": "⏰ 觸發時間", "value": timestamp, "inline": False},
                    {"name": "🔴 安全狀態", "value": "```diff\n- TRIGGERED\n```", "inline": False}
                ],
                "footer": {
                    "text": "NCKU CSIE AI Safety Monitor System • 2026"
                }
            }
        ]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code in [200, 204]:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔔 [Discord成功] 告警卡片已送達 Discord 頻道！")
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ [Discord失敗] HTTP 錯誤代碼: {response.status_code}")
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ [Discord錯誤] 連線失敗: {e}")

def send_discord_alert_async(timestamp, duration):
    """
    外部接口：採用非阻塞執行緒發送，確保主推論迴圈維持極高 FPS
    """
    thread = threading.Thread(target=_send_discord_worker, args=(timestamp, duration), daemon=True)
    thread.start()