"""Train the standalone YOLO11n + EQFL ablation on the 32-class mixdatasets split.

EQFL is the soft-quality-label variant of EFL. All other defaults below match the
recorded YOLO11n seed-0 baseline, so this run changes only classification loss.
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/root/ultralytics/mixdatasets_32classes/data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Use the same pretrained entry as the baseline.")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="/root/autodl-tmp/runs/train")
    parser.add_argument("--name", default="yolo11n_mixdatasets_32classes_eqfl_seed0")
    args = parser.parse_args()

    YOLO(args.weights).train(
        data=args.data,
        epochs=200,
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
        cls_loss="eqfl",
        efl_gamma=2.0,
        efl_scale=4.0,
        save=True,
        save_period=10,
        plots=True,
        exist_ok=False,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
