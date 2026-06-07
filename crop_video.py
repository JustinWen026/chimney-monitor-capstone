import os
import cv2

PROJECT_ROOT = r"D:\Programming\NCKU_Project_V2"
INPUT_VIDEO = os.path.join(PROJECT_ROOT, "test_video.mp4")      # 你剛剛合成好的 21 秒影片
OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "clean_video.mp4")    # 裁切後完美的乾淨影片

def main():
    print("[1/3] 正在讀取原始影片並計算裁切區域...")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"❌ 錯誤：找不到輸入影片 {INPUT_VIDEO}，請確認檔名！")
        return

    # 獲取影片原始參數
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"    -> 原始影片尺寸: {orig_width}x{orig_height} | FPS: {fps}")

    # ==============================================================================
    # 核心技術：動態矩陣裁剪配置 (ROI - Region of Interest)
    # ==============================================================================
    # 根據你上傳的截圖，我們採取以下黃金比例裁剪法：
    # 1. 頂部：切掉 5% (避開最上面的邊緣雜訊)
    # 2. 底部：切掉 12% (完美閹割掉 AI 內建的播放進度條)
    # 3. 兩側：左右各對稱切掉 5% (順便把右邊的 PixVerse 浮水印完全擼掉)
    
    y_start = int(orig_height * 0.05)
    y_end = int(orig_height * 0.88)
    x_start = int(orig_width * 0.05)
    x_end = int(orig_width * 0.95)
    
    # 計算裁切後的全新解析度
    new_width = x_end - x_start
    new_height = y_end - y_start
    print(f"    -> 🎯 裁切後全新尺寸: {new_width}x{new_height}")

    # 初始化影片寫入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (new_width, new_height))

    print("[2/3] 🚀 開始逐格進行矩陣裁剪...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用 Python Slice 語法直接對圖片矩陣進行高效裁切
        # 語法：frame[y_start:y_end, x_start:x_end]
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        
        # 寫入新影片
        out.write(cropped_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"    已處理 {frame_count} 個影格...")

    # 釋放資源
    cap.release()
    out.release()
    
    print("-" * 50)
    print(f"🎉 [3/3] 裁剪完成！乾淨版影片已誕生：")
    print(f"📂 輸出路徑: {OUTPUT_VIDEO}")
    print(f"💡 提示：現在你可以直接把 `mvp_pipeline.py` 裡的影片路徑改成這個 `clean_video.mp4` 了！")
    print("-" * 50)

if __name__ == "__main__":
    main()