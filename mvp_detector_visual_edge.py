import os
import time
import cv2
from datetime import datetime
from ultralytics import YOLO

# 匯入獨立拆分出去的 Discord 工安告警模組
from discord_notifier import send_discord_alert_async

# ==============================================================================
# 1. 系統參數配置
# ==============================================================================
PROJECT_ROOT        = r"D:\Programming\NCKU_Project_V2"
VIDEO_PATH          = os.path.join(PROJECT_ROOT, "clean_video_perfect.mp4")
MODEL_PATH          = os.path.join(PROJECT_ROOT, r"runs\detect\ncku_smoke_8to2_v2\weights\best.pt")
SCREENSHOT_DIR      = os.path.join(PROJECT_ROOT, "alerts")   # 截圖自動存放資料夾

ALERT_THRESHOLD_SEC = 3.0    # 狀態機安全閾值（秒）
ALERT_COOLDOWN_SEC  = 60.0   # 告警防連發冷卻時間（秒）

# 歸因對照（用於畫面疊加文字）
ATTRIBUTION_TEXT = {
    "black_smoky": "Root Cause: Combustion Ctrl Failure",
    "white_smoky": "Root Cause: Process Leakage / Steam Anomaly",
    "unknown":     "Root Cause: Pending Confirmation",
}

# ==============================================================================
# 2. 確保截圖資料夾存在
# ==============================================================================
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# ==============================================================================
# 3. 終端機輸出工具
# ==============================================================================
DIVIDER = "=" * 100

def print_banner():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(DIVIDER)
    print(f"  NCKU CSIE — AI Petrochemical Chimney Safety Monitor  |  啟動時間: {now}")
    print(f"  模型路徑 : {MODEL_PATH}")
    print(f"  影像來源 : {VIDEO_PATH}")
    print(f"  觸發閾值 : {ALERT_THRESHOLD_SEC}s   冷卻時間: {ALERT_COOLDOWN_SEC}s   截圖輸出: {SCREENSHOT_DIR}")
    print(DIVIDER)
    print()

def print_timing(t_pre, t_inf, t_post, t_state, t_draw, fps):
    """列印各階段詳細耗時（毫秒）"""
    total = t_pre + t_inf + t_post + t_state + t_draw
    print(
        f"\r  [PERF] "
        f"讀取:{t_pre:5.1f}ms  "
        f"推論:{t_inf:6.1f}ms  "
        f"後處理:{t_post:5.1f}ms  "
        f"狀態機:{t_state:5.2f}ms  "
        f"畫框:{t_draw:5.2f}ms  "
        f"| Total:{total:6.1f}ms  FPS:{fps:5.1f}   ",
        end="", flush=True
    )


# ==============================================================================
# 4. 截圖存檔
# ==============================================================================
def save_screenshot(frame, smoke_type: str) -> str:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alert_{smoke_type}_{ts}.jpg"
    path = os.path.join(SCREENSHOT_DIR, filename)
    cv2.imwrite(path, frame)
    return path


# ==============================================================================
# 5. OpenCV UI 繪製
# ==============================================================================
def draw_ui(frame, current_state, duration, fps, active_smoke_type, frame_counter):
    """
    將狀態機資訊繪製到畫面左上角。
    字型稍大但細（thickness=1），避免遮擋煙囪主體。
    """
    # 顏色對應
    if current_state == "NORMAL":
        status_color   = (0, 220, 0)
        progress_ratio = 0.0
    elif current_state == "ACCUMULATING":
        status_color   = (0, 165, 255)
        progress_ratio = min(duration / ALERT_THRESHOLD_SEC, 1.0)
    else:  # TRIGGERED
        status_color   = (0, 0, 220)
        progress_ratio = 1.0

    # ---------- 進度條 ----------
    bar_x, bar_y   = 10, 48
    bar_w, bar_h   = 160, 7
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
    fill_w = int(bar_w * progress_ratio)
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)

    # ---------- 文字 (thickness=1, 字號略大) ----------
    font    = cv2.FONT_HERSHEY_SIMPLEX
    thin    = 1
    cv2.putText(frame, f"STATE : {current_state}", (10, 35),
                font, 0.52, status_color, thin, cv2.LINE_AA)
    cv2.putText(frame, f"Time  : {duration:.1f}s / {ALERT_THRESHOLD_SEC}s", (10, 68),
                font, 0.40, (210, 210, 210), thin, cv2.LINE_AA)
    cv2.putText(frame, f"Smoke : {active_smoke_type if active_smoke_type else 'None'}", (10, 84),
                font, 0.40, (210, 210, 210), thin, cv2.LINE_AA)
    cv2.putText(frame, f"FPS   : {fps:.1f}", (10, 100),
                font, 0.40, (210, 210, 210), thin, cv2.LINE_AA)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), (10, 116),
                font, 0.38, (160, 160, 160), thin, cv2.LINE_AA)

    # ---------- TRIGGERED 額外顯示 ----------
    if current_state == "TRIGGERED":
        # 閃爍紅框（每 15 幀切換一次）
        if (frame_counter // 15) % 2 == 0:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)

        cv2.putText(frame, "[CRITICAL ALERT]", (10, 140),
                    font, 0.50, (0, 0, 220), thin, cv2.LINE_AA)

        # 歸因結果
        attr_text = ATTRIBUTION_TEXT.get(active_smoke_type or "unknown", ATTRIBUTION_TEXT["unknown"])
        cv2.putText(frame, attr_text, (10, 158),
                    font, 0.36, (0, 100, 255), thin, cv2.LINE_AA)

    return frame


# ==============================================================================
# 6. 主迴圈
# ==============================================================================
def main():
    print_banner()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在初始化 YOLOv11 推論引擎...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ 錯誤：無法開啟影片！")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms   = int(1000 / video_fps) if video_fps > 0 else 30
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 影片 FPS={video_fps:.1f}，每幀等待 {wait_ms}ms\n")

    # 狀態機核心變數
    current_state    = "NORMAL"
    smoke_start_time = None
    active_smoke_type = None
    duration          = 0.0
    last_alert_time   = 0.0

    # 顯示用 FPS 緩衝
    frame_counter   = 0
    display_fps     = 0.0

    print(DIVIDER)
    print("  🚀 推論迴圈啟動 — 按下 [Q] 結束程式")
    print(DIVIDER + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        current_time = time.time()

        # ── 推論 ──────────────────────────────────────────────────────────────
        results        = model(frame, device=0, verbose=False)
        annotated_frame = results[0].plot()

        t_pre  = results[0].speed.get('preprocess',  0.0)
        t_inf  = results[0].speed.get('inference',   0.0)
        t_post = results[0].speed.get('postprocess', 0.0)
        total_pipe = t_pre + t_inf + t_post

        # ── 解析偵測標籤 ──────────────────────────────────────────────────────
        detected_classes = [model.names[int(box.cls[0])] for box in results[0].boxes]
        has_smoke        = any(cls in ['white_smoky', 'black_smoky'] for cls in detected_classes)
        current_frame_smoke = next(
            (cls for cls in detected_classes if cls in ['white_smoky', 'black_smoky']), None
        )

        # ── FPS 計算（每 10 幀更新一次）────────────────────────────────────────
        if frame_counter % 10 == 0 or frame_counter == 1:
            display_fps = 1000.0 / total_pipe if total_pipe > 0 else 0.0

        # ── 狀態機演算法 ─────────────────────────────────────────────────────
        t_sm_start = time.perf_counter()

        if current_state == "NORMAL":
            duration = 0.0
            if has_smoke:
                current_state     = "ACCUMULATING"
                smoke_start_time  = current_time
                active_smoke_type = current_frame_smoke
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚨 狀態機跳轉: NORMAL ➔ ACCUMULATING  |  煙霧類型: {active_smoke_type}")
            else:
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] [STATE: NORMAL] 🟢 煙囪無異常，監控中...   ", end="", flush=True)

        elif current_state == "ACCUMULATING":
            if has_smoke:
                duration = current_time - smoke_start_time
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] [STATE: ACCUMULATING] ⏳ 蓄能中: {duration:.2f}s / {ALERT_THRESHOLD_SEC}s  |  煙霧: {active_smoke_type}   ", end="", flush=True)

                if duration >= ALERT_THRESHOLD_SEC:
                    current_state = "TRIGGERED"
                    readable_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    print(f"\n[{readable_time}] 🔥 演算法達到臨界值！狀態跳轉: TRIGGERED  |  累積時間: {duration:.2f}s  |  煙霧類型: {active_smoke_type}")

                    # 截圖存檔
                    screenshot_path = save_screenshot(annotated_frame, active_smoke_type or "unknown")
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📸 事件截圖已儲存: {screenshot_path}")

                    # Discord 告警（防連發）
                    if (current_time - last_alert_time) > ALERT_COOLDOWN_SEC:
                        send_discord_alert_async(
                            timestamp=readable_time,
                            duration=round(duration, 1),
                            smoke_type=active_smoke_type or "unknown",
                            screenshot_path=screenshot_path
                        )
                        last_alert_time = current_time
                    else:
                        remaining = ALERT_COOLDOWN_SEC - (current_time - last_alert_time)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🛡️  [防連發] 冷卻中，剩餘 {remaining:.0f}s，跳過本次 Discord 通報。")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 煙霧中斷，狀態機重置 NORMAL  |  蓄能時間: {duration:.2f}s（未達閾值）")
                current_state     = "NORMAL"
                smoke_start_time  = None
                active_smoke_type = None

        elif current_state == "TRIGGERED":
            if has_smoke:
                duration = current_time - smoke_start_time
                print(f"\r[{datetime.now().strftime('%H:%M:%S')}] [STATE: TRIGGERED] 🔴 警報持續！累計: {duration:.2f}s  |  煙霧: {active_smoke_type}   ", end="", flush=True)
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 工安事件解除，狀態機重置 NORMAL  |  最終持續時間: {duration:.2f}s")
                current_state     = "NORMAL"
                smoke_start_time  = None
                active_smoke_type = None

        t_sm_end = time.perf_counter()
        t_state_ms = (t_sm_end - t_sm_start) * 1000.0

        # ── OpenCV 畫框計時 ───────────────────────────────────────────────────
        t_draw_start = time.perf_counter()
        annotated_frame = draw_ui(
            annotated_frame, current_state, duration, display_fps,
            active_smoke_type, frame_counter
        )
        t_draw_end  = time.perf_counter()
        t_draw_ms   = (t_draw_end - t_draw_start) * 1000.0

        # ── 詳細耗時列印（每 10 幀，在 NORMAL 靜態行後輸出）──────────────────
        if frame_counter % 10 == 0:
            print_timing(t_pre, t_inf, t_post, t_state_ms, t_draw_ms, display_fps)

        # ── 顯示畫面 ──────────────────────────────────────────────────────────
        cv2.imshow("NCKU Petrochemical Chimney — AI Safety Monitor", annotated_frame)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print(f"\n\n[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 使用者手動結束程式。")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [EXIT] 資源已釋放，程式結束。\n")


if __name__ == "__main__":
    main()