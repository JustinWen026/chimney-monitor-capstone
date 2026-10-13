from pathlib import Path

from ultralytics import YOLO

if __name__ == '__main__':
    # 2. 改用 YOLOv11s (或 yolov8s.pt / yolov9c.pt，格式都通用)
    model = YOLO('yolo11s.pt') 

    model.train(
        # 使用你剛才確認的絕對路徑
        data=str(Path(__file__).resolve().parent / 'datasets' / 'data.yaml'),
        
        # 3. 學長要求：1000 輪 + Early Stopping
        epochs=1000,       
        patience=50,       
        
        # 7. imgsz 拉高到 1024 (對付小煙囪)
        imgsz=1024,        
        
        # 6. 助教要求的小指標 (學習率)
        lr0=0.00001,       
        
        # 5060 效能分配
        batch=8,           # 如果爆顯存 (OOM) 就改 4
        device=0,          
        name='ncku_smoke_8to2_v2',
        augment=True,      # 4. 減少誤判
        workers=4          # 加速資料讀取
    )