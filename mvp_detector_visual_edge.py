import os
import time
import cv2
from datetime import datetime
from ultralytics import YOLO

# 匯入我們獨立拆分出去的 Discord 告警模組
from discord_notifier import send_discord_alert_async

# ==============================================================================
# 1. 系統參數配置
# ==============================================================================
PROJECT_ROOT = r"D:\Programming\NCKU_Project_V2"
VIDEO_PATH = os.path.join(PROJECT_ROOT, "clean_video_perfect.mp4")
MODEL_PATH = os.path.join(PROJECT_ROOT, r"runs\detect\ncku_smoke_8to2_v2\weights\best.pt")

ALERT_THRESHOLD_SEC = 3.0   # 狀態機安全閾值
ALERT_COOLDOWN_SEC = 60.0   # 告警防連發冷卻時間 (秒)

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在初始化 YOLOv11 推論引擎...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ 錯誤：無法開啟影片！")
        return

    # 狀態機與告警時序核心變數
    current_state = "NORMAL"
    smoke_start_time = None
    active_smoke_type = None
    duration = 0.0
    last_alert_time = 0.0  # 記錄上一次成功觸發 Discord 告警的時間戳 (time.time())

    # FPS 平滑緩衝
    frame_counter = 0
    display_pipe_fps = 0.0

    print("\n" + "="*95)
    print(" 🚀 演算法模組視覺強化整合版後台控制台已啟動...")
    print(" 說明：已將時序狀態機演算法的蓄能過程與 Discord 解耦主動通報模組完美對接。")
    print("="*95 + "\n")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000 / video_fps) if video_fps > 0 else 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # GPU 推論 (RTX 5060 加速)
        results = model(frame, device=0, verbose=False)
        annotated_frame = results[0].plot()
        
        # FPS 計算
        t_pre  = results[0].speed.get('preprocess', 0.0)
        t_inf  = results[0].speed.get('inference', 0.0)
        t_post = results[0].speed.get('postprocess', 0.0)
        total_latency = t_pre + t_inf + t_post

        frame_counter += 1
        if frame_counter % 10 == 0 or frame_counter == 1:
            display_pipe_fps = 1000.0 / total_latency if total_latency > 0 else 0.0

        # 解析煙霧標籤
        detected_classes = [model.names[int(box.cls[0])] for box in results[0].boxes]
        has_smoke = any(cls in ['white_smoky', 'black_smoky'] for cls in detected_classes)
        current_frame_smoke = next((cls for cls in detected_classes if cls in ['white_smoky', 'black_smoky']), None)

        current_time = time.time()

        # ==============================================================================
        # ⚙️ 時序狀態機演算法內核與運算 (嵌入解耦通報與防連發機制)
        # ==============================================================================
        if current_state == "NORMAL":
            duration = 0.0
            if has_smoke:
                current_state = "ACCUMULATING"
                smoke_start_time = current_time
                active_smoke_type = current_frame_smoke
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚨 狀態機跳轉: NORMAL ➔ ACCUMULATING")
            else:
                print(f"\r[STATE: NORMAL] 🟢 煙囪無異常，狀態機守衛中... | 效能: {display_pipe_fps:.1f} FPS   ", end="", flush=True)

        elif current_state == "ACCUMULATING":
            if has_smoke:
                duration = current_time - smoke_start_time
                print(f"\r[STATE: ACCUMULATING] ⏳ 演算法時間蓄能中: {duration:.2f}s / {ALERT_THRESHOLD_SEC}s... ", end="", flush=True)
                
                if duration >= ALERT_THRESHOLD_SEC:
                    # 關鍵瞬間：時序滿 3 秒，首度衝破臨界點！
                    current_state = "TRIGGERED"
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔥 演算法達到滿載臨界值！狀態跳轉: TRIGGERED")
                    
                    # 💡 安全通報防連發邏輯
                    if (current_time - last_alert_time) > ALERT_COOLDOWN_SEC:
                        readable_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        # 呼叫外部解耦模組，透過非阻塞執行緒非同步發信
                        send_discord_alert_async(timestamp=readable_time, duration=round(duration, 1))
                        last_alert_time = current_time
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🛡️  [防連發保護] 處於冷卻期內，跳過本次 Discord 卡片發送。")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 異常中斷，狀態機重置回 NORMAL")
                current_state = "NORMAL"
                smoke_start_time = None

        elif current_state == "TRIGGERED":
            if has_smoke:
                duration = current_time - smoke_start_time
                print(f"\r[STATE: TRIGGERED] 🔴 警報持續爆發中！累計異常時間: {duration:.2f}s   ", end="", flush=True)
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 工安危機解除，狀態機重置")
                current_state = "NORMAL"
                smoke_start_time = None

        # ==============================================================================
        # 🎨 黑科技：將狀態機演算法視覺化 (UI 精簡與無亂碼版)
        # ==============================================================================
        # 將 UI 移至左上角並大幅縮小，避免擋住畫面主體
        bar_x, bar_y = 10, 40
        bar_width, bar_height = 120, 8
        
        if current_state == "NORMAL":
            status_color = (0, 255, 0)     # 綠色
            progress_ratio = 0.0
        elif current_state == "ACCUMULATING":
            status_color = (0, 165, 255)   # 橘黃色
            progress_ratio = min(duration / ALERT_THRESHOLD_SEC, 1.0)
        else: # TRIGGERED
            status_color = (0, 0, 255)     # 鮮紅色
            progress_ratio = 1.0

        # 1. 繪製進度條外框
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)
        # 2. 依據時序演算法的累積時間，動態填滿進度條
        fill_width = int(bar_width * progress_ratio)
        if fill_width > 0:
            cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), status_color, -1)
        
        # 3. 疊加文字說明 (縮小字體尺寸與間距，全面改用英文避免 OpenCV 亂碼)
        cv2.putText(annotated_frame, f"STATE: {current_state}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Time: {duration:.1f}s / {ALERT_THRESHOLD_SEC}s", (10, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"FPS: {display_pipe_fps:.1f}", (10, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # 4. 如果進入 TRIGGERED 狀態，顯示緊湊的英文告警
        if current_state == "TRIGGERED":
            cv2.putText(annotated_frame, "[CRITICAL ALERT]", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("NCKU Petrochemical Chimney - Advanced Algorithmic Dashboard", annotated_frame)

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()