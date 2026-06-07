import os
import cv2

PROJECT_ROOT = r"D:\Programming\NCKU_Project_V2"
IMAGE_PATH = os.path.join(PROJECT_ROOT, "no_smoke.jpg")       # 你的無煙照片
KLING_VIDEO_PATH = os.path.join(PROJECT_ROOT, "kling_output.mp4") # 可靈生成的11秒影片
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "test_video.mp4")       # 最終合成的完美影片

def main():
    print("[1/4] 正在讀取可靈影片以獲取格式參數...")
    kling_cap = cv2.VideoCapture(KLING_VIDEO_PATH)
    if not kling_cap.isOpened():
        print("❌ 錯誤：找不到可靈生成的影片，請檢查路徑或檔名！")
        return
        
    # 讀取可靈影片的影片參數 (確保合成影片的解析度與 FPS 完全一致)
    fps = kling_cap.get(cv2.CAP_PROP_FPS)
    width = int(kling_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(kling_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kling_cap.release()
    
    print(f"    -> 偵測到影片參數: {width}x{height}, FPS: {fps}")

    print("[2/4] 正在讀取無煙照片並調整尺寸...")
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("❌ 錯誤：找不到無煙照片 no_smoke.jpg！")
        return
    # 強制將照片縮放到跟影片一樣大
    img_resized = cv2.resize(img, (width, height))

    # 初始化影片寫入器 (使用 MP4V 編碼，Windows 支援度最高)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # 產生 10 秒的無煙片段
    # 總影格數 = 10 秒 * 每秒幾格 (FPS)
    no_smoke_seconds = 10
    total_no_smoke_frames = int(no_smoke_seconds * fps)
    print(f"[3/4] 正在人工生成 {no_smoke_seconds} 秒的「完全無煙靜態片段」({total_no_smoke_frames} 格)...")
    for _ in range(total_no_smoke_frames):
        out.write(img_resized)

    # 拼接可靈的 11 秒影片
    print("[4/4] 正在將可靈的冒煙影片拼接到尾端...")
    kling_cap = cv2.VideoCapture(KLING_VIDEO_PATH)
    while kling_cap.isOpened():
        ret, frame = kling_cap.read()
        if not ret:
            break
        # 確保影格尺寸一致後寫入
        frame_resized = cv2.resize(frame, (width, height))
        out.write(frame_resized)
        
    kling_cap.release()
    out.release()
    print(f"\n🎉 恭喜！神級測試素材 `test_video.mp4` 已完美合體！")
    print(f"總長度：約 21 秒 (前 10 秒無煙 ➔ 第 10.5 秒開始暴煙)")
    print(f"檔案路徑: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()