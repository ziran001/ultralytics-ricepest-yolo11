from ultralytics import YOLO


model = YOLO("ultralytics/cfg/models/11/yolo11-bafe.yaml")
model.load("yolo11n.pt")

model.train(
    data="/root/ultralytics-8.3.27/mixed_dataset/data.yaml",
    epochs=300,
    imgsz=640,
    batch=32,
    device=0,
    workers=32,
    lr0=0.002,
    optimizer="auto",
    patience=100,
    save=True,
    project="runs/train",
    name="yolo11_bafe_pretrained_mixed_dataset",
)
