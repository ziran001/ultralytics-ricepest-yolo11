from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='/root/ultralytics-8.3.27/mixed_dataset 1+1/data.yaml',
    epochs=300,
    imgsz=640,
    batch=32,
    device=0,
    workers=32,
    lr0=0.002,
    optimizer='auto',
    patience=100,
    save=True,
    project='runs/train',
    name='yolov8'
)