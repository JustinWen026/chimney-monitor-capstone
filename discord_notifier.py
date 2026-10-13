import os
import requests
import threading
import json
from datetime import datetime
from pathlib import Path

# =====================================================================
# Discord Webhook 設定區
# =====================================================================
# 從環境變數讀取，避免把 Webhook 網址寫進程式碼
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "YOUR_DISCORD_WEBHOOK_URL_HERE")

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

    # ── 判斷截圖是否可上傳 ────────────────────────────────────────────
    has_screenshot = (
        screenshot_path is not None
        and Path(screenshot_path).exists()
        and Path(screenshot_path).stat().st_size > 0
    )

    # embed 裡的圖片欄位：有截圖就用 attachment://，否則顯示路徑文字
    if has_screenshot:
        image_field_value = "（截圖已附於本則訊息下方）"
        embed_image = {"url": "attachment://alert_screenshot.png"}
    else:
        image_field_value = f"`{screenshot_path}`" if screenshot_path else "（未附圖）"
        embed_image = None

    embed = {
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
            {"name": "🖼 事件截圖",   "value": image_field_value,                         "inline": False},
            {
                "name":  "⚙️ 系統狀態",
                "value": "```\nSTATUS : TRIGGERED\nSOURCE : YOLOv11 + Temporal FSM + Temporal Onset Priority\n```",
                "inline": False,
            },
        ],
        "footer": {
            "text": "NCKU CSIE — AI Safety Monitor System v7 • 本通報由自動化系統產生，請人員現場複核後執行處置"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    if embed_image:
        embed["image"] = embed_image

    payload = {
        "username": "NCKU 石化園區安全監控系統",
        "avatar_url": "https://i.imgur.com/8nS47H3.png",
        "embeds": [embed],
    }

    try:
        if has_screenshot:
            # ── 有截圖：用 multipart/form-data 同時上傳圖片與 embed ──
            with open(screenshot_path, "rb") as img_file:
                response = requests.post(
                    DISCORD_WEBHOOK_URL,
                    data={"payload_json": json.dumps(payload)},
                    files={"files[0]": ("alert_screenshot.png", img_file, "image/png")},
                    timeout=10,
                )
        else:
            # ── 無截圖：純 JSON ───────────────────────────────────────
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)

        if response.status_code in [200, 204]:
            tag = "含截圖" if has_screenshot else "無截圖"
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔔 [Discord 成功({tag})] 工安告警通報已送達頻道。")
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
    """非同步發送 Discord 告警，不阻塞主推論迴圈。
    
    screenshot_path 若傳入有效的 .png 路徑，會直接將圖片附加在 Discord 訊息內。
    """
    thread = threading.Thread(
        target=_send_discord_worker,
        args=(timestamp, duration, smoke_type, chimney_id, density, confidence, screenshot_path),
        daemon=True,
    )
    thread.start()