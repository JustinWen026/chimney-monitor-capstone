import requests
import threading
from datetime import datetime

# =====================================================================
# Discord Webhook 設定區
# =====================================================================
DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL_HERE"

ATTRIBUTION_MAP = {
    "black_smoky": {
        "cause":  "燃燒控制系統異常（不完全燃燒）",
        "level":  "Level 3 — 高風險",
        "action": "立即通報設備維護組，暫停相關製程，進行鍋爐/燃燒器緊急檢修",
    },
    "white_smoky": {
        "cause":  "製程洩漏或蒸氣溢散異常",
        "level":  "Level 2 — 中風險",
        "action": "啟動洩漏排查程序，確認管線接頭與閥件密封狀態",
    },
    "unknown": {
        "cause":  "煙霧類型待確認",
        "level":  "Level 2 — 中風險",
        "action": "人員現場確認，待視覺特徵穩定後重新歸因",
    },
}


def _send_discord_worker(
    timestamp: str,
    duration: float,
    smoke_type: str,
    chimney_id: str,
    density: float,
    confidence: float,
    screenshot_path: str | None = None,
):
    if DISCORD_WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚠️  [Discord] 尚未設定 Webhook URL，跳過發送。")
        return

    attr = ATTRIBUTION_MAP.get(smoke_type, ATTRIBUTION_MAP["unknown"])
    smoke_label = {
        "black_smoky": "黑煙 (Black Smoky)",
        "white_smoky": "白煙 (White Smoky)",
    }.get(smoke_type, "待確認")

    screenshot_note = f"`{screenshot_path}`" if screenshot_path else "（未附圖）"

    payload = {
        "username": "NCKU 石化園區安全監控系統",
        "avatar_url": "https://i.imgur.com/8nS47H3.png",
        "embeds": [
            {
                "title": "⚠️ 工業安全告警通報：偵測到煙囪異常排煙",
                "description": (
                    "自動監控系統偵測到廠區煙囪排煙持續超過安全閾值。\n"
                    "系統已完成初步歸因判定，請相關人員依處置建議立即因應。"
                ),
                "color": 13632027,
                "fields": [
                    {"name": "🗓 告警時間",   "value": timestamp,                                 "inline": True},
                    {"name": "⏱ 持續時間",   "value": f"{duration} 秒（門檻：3.0 秒）",          "inline": True},
                    {"name": "🏭 事件煙囪",   "value": chimney_id,                                "inline": True},
                    {"name": "💨 煙霧類型",   "value": smoke_label,                               "inline": True},
                    {"name": "📊 偵測密度",   "value": f"{density:.1%}（滑動視窗命中率）",        "inline": True},
                    {"name": "🎯 偵測信心度", "value": f"{confidence:.2f}",                       "inline": True},
                    {"name": "🔍 歸因判定",   "value": attr["cause"],                             "inline": False},
                    {"name": "🚦 風險等級",   "value": attr["level"],                             "inline": True},
                    {"name": "📋 建議處置",   "value": attr["action"],                            "inline": False},
                    {"name": "🖼 事件截圖",   "value": screenshot_note,                           "inline": False},
                    {
                        "name":  "⚙️ 系統狀態",
                        "value": "```\nSTATUS : TRIGGERED\nSOURCE : YOLOv11 + Temporal FSM + Emission Zone Attribution\n```",
                        "inline": False,
                    },
                ],
                "footer": {
                    "text": "NCKU CSIE — AI Safety Monitor System v5 • 本通報由自動化系統產生，請人員現場複核後執行處置"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        ],
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        if response.status_code in [200, 204]:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔔 [Discord 成功] 工安告警通報已送達頻道。")
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ [Discord 失敗] HTTP 狀態碼: {response.status_code}")
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ❌ [Discord 錯誤] 連線失敗: {e}")


def send_discord_alert_async(
    timestamp: str,
    duration: float,
    smoke_type: str = "unknown",
    chimney_id: str = "Unknown",
    density: float = 0.0,
    confidence: float = 0.0,
    screenshot_path: str | None = None,
):
    """
    外部接口：非阻塞執行緒發送。

    Args:
        timestamp      : 觸發時間（可讀格式）
        duration       : 累積異常秒數
        smoke_type     : 'black_smoky' / 'white_smoky' / 'unknown'
        chimney_id     : 觸發告警的煙囪穩定識別編號
        density        : 滑動視窗偵測密度（0.0~1.0）
        confidence     : YOLO 最高信心度
        screenshot_path: 事件截圖路徑（選填）
    """
    thread = threading.Thread(
        target=_send_discord_worker,
        args=(timestamp, duration, smoke_type, chimney_id, density, confidence, screenshot_path),
        daemon=True,
    )
    thread.start()