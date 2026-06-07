import os
import time
import cv2
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
# 1. 系統參數配置
# ==============================================================================
PROJECT_ROOT = r"D:\Programming\NCKU_Project_V2"
# 讀取你剛剛切好的完美無水印影片
VIDEO_PATH = os.path.join(PROJECT_ROOT, "clean_video_perfect.mp4")
# 你的 YOLOv11 最佳權重路徑
MODEL_PATH = os.path.join(PROJECT_ROOT, r"runs\detect\ncku_smoke_8to2_v2\weights\best.pt")

# 時序狀態機閾值設定
ALERT_THRESHOLD_SEC = 3.0   # 當煙霧持續存在超過 3 秒，狀態切換為 TRIGGERED

# ==============================================================================
# 2. 主程式初始化與即時迴圈
# ==============================================================================
def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在初始化 YOLOv11 推論引擎...")
    model = YOLO(MODEL_PATH)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INIT] 正在載入無雜訊測試影片: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ 錯誤：無法開啟影片，請檢查影片路徑與檔名是否正確！")
        return

    # 狀態機變數初始化
    # 狀態包含: "NORMAL" (安全), "ACCUMULATING" (蓄能中), "TRIGGERED" (已達閾值告警)
    current_state = "NORMAL"
    smoke_start_time = None
    active_smoke_type = None

    print("\n" + "="*70)
    print(" 🚀 石化園區異常煙霧監控系統後台已啟動...")
    print(" 提示：請將 OpenCV 視窗拖移至螢幕右側，VS Code 擺在左側，即可完成黃金Demo配置。")
    print(" 結束方式：在 OpenCV 畫面上按下 'q' 鍵即可安全退出。")
    print("="*70 + "\n")

    # 取得影片的 FPS，用來動態調節播放速度
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000 / fps) if fps > 0 else 30

    while cap.isOpened():
        start_loop_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [INFO] 影片播放完畢。")
            break

        # 核心技術：呼叫 RTX 5060 進行加速推論 (device=0)
        results = model(frame, device=0, verbose=False)
        
        # 讓 YOLO 自動幫我們把預測的框 (Bounding Box) 畫在畫面上
        annotated_frame = results[0].plot()
        
        # 解析當前這一格畫面中被偵測到的類別
        detected_classes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            detected_classes.append(cls_name)
            
        # 判定當前影格是否存在你模型定義的煙霧標籤
        # 這裡會自動匹配你的 'white_smoky' 或 'black_smoky'
        has_smoke = any(cls in ['white_smoky', 'black_smoky'] for cls in detected_classes)
        current_frame_smoke = next((cls for cls in detected_classes if cls in ['white_smoky', 'black_smoky']), None)

        current_time = time.time()

        # ==============================================================================
        # 核心演算法：時序狀態機邏輯與 Terminal 炫酷日誌輸出
        # ==============================================================================
        if current_state == "NORMAL":
            if has_smoke:
                # 轉移至「蓄能狀態」
                current_state = "ACCUMULATING"
                smoke_start_time = current_time
                active_smoke_type = current_frame_smoke
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [STATE CHANGE] 🟢 NORMAL ➔ 🟡 ACCUMULATING")
                print(f"    [DETECTED] 偵測到煙囪開始排放: {active_smoke_type}")
            else:
                # 沒煙時在終端機刷出綠色心跳包，證明系統存活且正常運作
                print(f"\r[NORMAL] 煙囪排放監測中... 狀態安全 (0.00s)", end="", flush=True)

        elif current_state == "ACCUMULATING":
            if has_smoke:
                duration = current_time - smoke_start_time
                print(f"\r[ACCUMULATING] ⏳ 煙霧持續累積中: {duration:.2f}秒 / 閾值 {ALERT_THRESHOLD_SEC}秒...", end="", flush=True)
                
                if duration >= ALERT_THRESHOLD_SEC:
                    current_state = "TRIGGERED"
                    print("\n" + "🔥" * 30)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] [CRITICAL ALERT] ⚠️ 警報觸發！ SYSTEM STATE: TRIGGERED")
                    print(f"    [REASON] 異常煙霧 ({active_smoke_type}) 連續排放已達 {duration:.2f} 秒！")
                    print(f"    [ACTION] 通訊應用端模組已接管，即將發送主動式工安事故分析報告...")
                    print("🔥" * 30)
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [STATE CHANGE] 🟡 ACCUMULATING ➔ 🟢 NORMAL (煙霧屬偶發波動，警報解除)")
                current_state = "NORMAL"
                smoke_start_time = None
                active_smoke_type = None

        elif current_state == "TRIGGERED":
            if has_smoke:
                # 雖然已經觸發，但只要煙霧還在，後台就持續噴出高危日誌
                duration = current_time - smoke_start_time
                print(f"\r[TRIGGERED] 🚨 煙囪仍持續處於異常排煙狀態！已累計時間: {duration:.2f}秒", end="", flush=True)
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [STATE CHANGE] 🔴 TRIGGERED ➔ 🟢 NORMAL (異常煙霧已消失，環境恢復安全)")
                current_state = "NORMAL"
                smoke_start_time = None
                active_smoke_type = None

        # ==============================================================================
        # 3. OpenCV 視窗 UI 疊加與渲染
        # ==============================================================================
        # 依據不同狀態給予視窗左上角不同的顏色提示（綠/黃/紅）
        ui_color = (0, 0, 255) if current_state == "TRIGGERED" else (0, 255, 255) if current_state == "ACCUMULATING" else (0, 255, 0)
        
        cv2.putText(annotated_frame, f"SYSTEM STATE: {current_state}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ui_color, 2, cv2.LINE_AA)
        
        # 命名視窗，並讓它允許使用者手動縮放調整
        window_name = "NCKU Petrochemical Zone - AI Chimney Smoke Monitoring MVP"
        cv2.imshow(window_name, annotated_frame)

        # 按 'q' 鍵安全退出
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [INFO] 使用者手動中止系統。")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] 後台推論引擎安全關閉。")

if __name__ == "__main__":
    main()