from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/root/ultralytics-8.3.27/runs/train/yolo11_mixed_dataset/weights/best.pt")

    # YOLO 官方标准写法，绝对正确
    model.val(
        data="/root/ultralytics-8.3.27/mixed_dataset/data.yaml",
        split="val",      # 用 test 集
        device=0,
        conf=0.001,
        iou=0.6
    )