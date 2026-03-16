from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os

# 1. 載入模型 (YOLOv11 輕量版)
model = YOLO('yolo11n.pt')

# 2. 定義你的圖片路徑 (請確認這兩個檔案都在同一個資料夾下)
img_list = ['smoke1.png', 'smoke2.png'] 

plt.figure(figsize=(15, 10))
plt.suptitle("Initial Inference Analysis on Factory Smoke Dataset", fontsize=16)

for i, img_path in enumerate(img_list):
    if not os.path.exists(img_path):
        print(f"❌ 找不到檔案: {img_path}，請檢查檔名是否正確！")
        continue
        
    # 執行偵測
    # results = model(img_path)
    # 改成這樣 (加入 conf=0.01，代表 1% 信心就顯示)
    results = model(img_path, conf=0.01)
    # 取得畫好框的結果圖
    res_plotted = results[0].plot()
    
    # 畫在左邊或右邊 (第一列)
    plt.subplot(2, 2, i+1)
    plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    plt.title(f"Detection Result: {img_path}")
    plt.axis('off')
    
    # 下方畫出信心值統計 (第二列)
    plt.subplot(2, 2, i+3)
    confs = [float(box.conf[0]) for box in results[0].boxes]
    
    if confs:
        plt.bar(range(len(confs)), confs, color='skyblue')
        plt.ylabel("Confidence Score")
    else:
        plt.text(0.5, 0.5, "No Objects Detected", ha='center')
        
    plt.ylim(0, 1)
    plt.title(f"Confidence Analysis: {img_path}")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('final_report.png')
print("\n" + "="*30)
print("✅ 恭喜！對比報告 'final_report.png' 已生成！")
print("="*30)