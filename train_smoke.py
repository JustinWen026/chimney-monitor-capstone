import os
os.environ['TORCH_CUDA_ARCH_LIST'] = "12.0"  # RTX 5060 Blackwell 架構
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    model.train(
        data='D:/Programming/NCKU_Project/Smoke_Detection/datasets/data.yaml',
        epochs=3,
        imgsz=640,
        batch=8,
        device=0,
        name='ncku_smoke_v1'
    )