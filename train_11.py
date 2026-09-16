"""Train the migrated YOLO11 rice-pest model."""

from ultralytics import YOLO


if __name__ == "__main__":
    # Custom YOLO11 route: MEN(P3) + Weighted-P3Fusion.
    model = YOLO("ultralytics/cfg/models/11/yolo11-men-p3-weighted-p3fusion.yaml")

    # If a local YOLO11 pretrained checkpoint is available, load matching layers before training:
    # model.load("/path/to/yolo11n.pt")
    results = model.train(
        data="/root/ultralytics/mixdatasets_32classes/data.yaml",
        epochs=200,
        patience=40,
        imgsz=1024,
        batch=32,
        device=0,
        workers=8,
        amp=True,
        save=True,
        save_period=10,
        project="runs/train",
        name="yolo11_men_p3_weighted_p3fusion_mixdatasets_32classes_1024",
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.0,
    )

    print(f"Training results saved to: {model.trainer.save_dir}")

