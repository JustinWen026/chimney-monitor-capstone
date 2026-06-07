import os
import cv2

PROJECT_ROOT = r"D:\Programming\NCKU_Project_V2"
# 輸入：你合成好的 21 秒原始影片 (含浮水印和 AI 進度條)
INPUT_VIDEO = os.path.join(PROJECT_ROOT, "test_video.mp4")      
# 輸出：完美乾淨的影片
OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "clean_video_perfect.mp4")

def main():
    print("[1/3] 正在讀取原始影片並進行專業裁剪...")
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
    # 核心技術：完美的矩陣裁剪配置 (大幅提高裁剪比例)
    # ==============================================================================
    # 根據 user 上傳的 image_0.png 成果進行大幅修正：
    # 1. 頂部：維持不切或小切 (避免影響天空煙霧) -> start 0.02
    # 2. 底部：大幅提高裁剪比例，徹底擼掉橘色進度條 -> end 0.80 (切掉 bottom 20%)
    # 3. 左側：維持小切 (不影響煙囪) -> start 0.05
    # 4. 右側：大幅提高裁剪比例，徹底閹割掉 PixVerse 浮水印 -> end 0.85 (切掉 right 15%)
    
    y_start = int(orig_height * 0.02)
    y_end = int(orig_height * 0.80)   # 大切底部
    x_start = int(orig_width * 0.05)
    x_end = int(orig_width * 0.75)    # 大切右側
    
    # 計算裁切後的全新解析度
    new_width = x_end - x_start
    new_height = y_end - y_start
    print(f"    -> 🎯 修正後全新解析度: {new_width}x{new_height}")

    # 初始化影片寫入器 (使用 MP4V 編碼)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (new_width, new_height))

    print("[2/3] 🚀 正在逐格進行矩陣裁剪 (約需 10~20 秒)...")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 使用 Python Slice 語法直接對圖片矩陣進行裁剪
        cropped_frame = frame[y_start:y_end, x_start:x_end]
        
        # 寫入新影片
        out.write(cropped_frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"    已處理 {frame_count} 個影格...")

    # 釋放資源
    cap.release()
    out.release()
    
    print("-" * 60)
    print(f"🎉 [3/3] 修正完成！完美乾淨影片已誕生：")
    print(f"📂 輸出路徑: {OUTPUT_VIDEO}")
    print(f"💡 提示：請在 `test_yolo_video.py` 中將影片路徑更新為這個完美的檔案。")
    print("-" * 60)

if __name__ == "__main__":
    main()