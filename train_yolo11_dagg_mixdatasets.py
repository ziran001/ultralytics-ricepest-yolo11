"""Train the standalone YOLO11n DAGG(P3) experiment on mixdatasets_32classes."""

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
# Ultralytics resolves this n-scale alias to yolo11-dagg-p3.yaml and records scale=n.
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/11/yolo11n-dagg-p3.yaml"


def parse_args():
    """Parse paths and hardware overrides while keeping the experiment defaults fixed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/root/ultralytics/mixdatasets_32classes/data.yaml")
    parser.add_argument("--weights", default="/root/ultralytics/yolo11n.pt")
    parser.add_argument("--project", default="/root/autodl-tmp/runs/train")
    parser.add_argument("--name", default="yolo11n_mixdatasets_32classes_dagg_p3_g025_s025_seed0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main():
    """Build DAGG from YAML, transfer YOLO11n weights, and run the controlled experiment."""
    args = parse_args()
    model = YOLO(str(DEFAULT_MODEL)).load(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=40,
        imgsz=1024,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        amp=True,
        seed=0,
        deterministic=True,
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.0,
        close_mosaic=10,
        save=True,
        save_period=10,
        plots=False,
        project=args.project,
        name=args.name,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
