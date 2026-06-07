import os
import time
import cv2
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
# 1. 系統參數配置
# ==============================================================================
PROJECT_ROOT = r"D:\Programming\NCKU_Project_V2"
VIDEO_PATH = os.path.join(PROJECT_ROOT, "clean_video_perfect.mp4")
MODEL_PATH = os.path.join(PROJECT_ROOT, r"runs\detect\ncku_smoke_8to2_v2\weights\best.pt")

ALERT_THRESHOLD_SEC = 3.0   # 狀態機閾值

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在初始化 YOLOv11 推論引擎...")
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ 錯誤：無法開啟影片！")
        return

    # 狀態機變數
    current_state = "NORMAL"
    smoke_start_time = None
    active_smoke_type = None

    # FPS 平滑緩衝變數
    frame_counter = 0
    display_pure_fps = 0.0
    display_pipe_fps = 0.0
    display_latency = 0.0

    print("\n" + "="*95)
    print(" 🚀 石化園區監控系統 - 工業級即時控制台 (後台平滑化儀表板已啟動)")
    print(" 說明：下方狀態列採用單行刷新技術。在未觸發工安事件前，日誌將保持平穩跳動。")
    print("="*95 + "\n")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000 / video_fps) if video_fps > 0 else 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [INFO] 影片播放完畢。")
            break

        # GPU 核心推論
        results = model(frame, device=0, verbose=False)
        annotated_frame = results[0].plot()
        
        # 獲取時間指標 (ms)
        t_pre  = results[0].speed.get('preprocess', 0.0)
        t_inf  = results[0].speed.get('inference', 0.0)
        t_post = results[0].speed.get('postprocess', 0.0)
        total_latency = t_pre + t_inf + t_post

        # ----------------------------------------------------------------------
        # 📊 緩衝平滑機制：每 10 格影格才更新一次數值，防止數值閃爍閃爍太快導致觀眾看不清
        # ----------------------------------------------------------------------
        frame_counter += 1
        if frame_counter % 10 == 0 or frame_counter == 1:
            display_pure_fps = 1000.0 / t_inf if t_inf > 0 else 0.0
            display_pipe_fps = 1000.0 / total_latency if total_latency > 0 else 0.0
            display_latency = total_latency

        # 解析煙霧狀態
        detected_classes = [model.names[int(box.cls[0])] for box in results[0].boxes]
        has_smoke = any(cls in ['white_smoky', 'black_smoky'] for cls in detected_classes)
        current_frame_smoke = next((cls for cls in detected_classes if cls in ['white_smoky', 'black_smoky']), None)

        current_time = time.time()

        # 構建炫酷儀表板狀態字串
        dashboard_str = (
            f"[硬體延遲: {display_latency:.1f}ms] "
            f"➔ 🔋 核心算力: {display_pure_fps:.1f} FPS | "
            f"📊 系統吞吐: {display_pipe_fps:.1f} FPS"
        )

        # ==============================================================================
        # 核心技術：時序狀態機演算法
        # ==============================================================================
        if current_state == "NORMAL":
            if has_smoke:
                current_state = "ACCUMULATING"
                smoke_start_time = current_time
                active_smoke_type = current_frame_smoke
                # 狀態跳轉時，破行印出關鍵日誌，拉出層次感
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🚨 [狀態變更] 🟢 NORMAL ➔ 🟡 ACCUMULATING")
                print(f"    [EVENT] 偵測到煙囪開始排放異常煙霧: {active_smoke_type}")
            else:
                # \r 搭配 end="" 讓這一行原地刷新，觀眾看起來像動態儀表板，完全不洗畫面！
                print(f"\r[STATUS] 🟢 系統運作安全 || {dashboard_str}      ", end="", flush=True)

        elif current_state == "ACCUMULATING":
            if has_smoke:
                duration = current_time - smoke_start_time
                print(f"\r[STATUS] ⏳ 工安事件蓄能中: {duration:.2f}秒 / 3.00秒 || {dashboard_str}      ", end="", flush=True)
                
                if duration >= ALERT_THRESHOLD_SEC:
                    current_state = "TRIGGERED"
                    print("\n" + "🔥" * 50)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🛑 [CRITICAL ALERT] SYSTEM STATE: TRIGGERED")
                    print(f"    [REASON] 煙囪異常排煙已持續 {duration:.2f} 秒，達到工安警戒閾值！")
                    print(f"    [BENCHMARK] {dashboard_str}")
                    print(f"    [ACTION] 已掛載通訊模組，觸發主動式即時告警機制...")
                    print("🔥" * 50)
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 [警報解除] 🟡 ACCUMULATING ➔ 🟢 NORMAL")
                current_state = "NORMAL"
                smoke_start_time = None

        elif current_state == "TRIGGERED":
            if has_smoke:
                duration = current_time - smoke_start_time
                print(f"\r[STATUS] 🔴 系統處於 TRIGGERED 警報狀態！已持續: {duration:.2f}秒 || {dashboard_str}      ", end="", flush=True)
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🟢 [工安恢復] 🔴 TRIGGERED ➔ 🟢 NORMAL")
                current_state = "NORMAL"
                smoke_start_time = None

        # OpenCV 畫面疊加
        cv2.putText(annotated_frame, f"STATE: {current_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if current_state == "TRIGGERED" else (0,255,255) if current_state == "ACCUMULATING" else (0,255,0), 2)
        cv2.putText(annotated_frame, f"System: {display_pipe_fps:.1f} FPS", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        cv2.imshow("NCKU Petrochemical Zone MVP", annotated_frame)

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()