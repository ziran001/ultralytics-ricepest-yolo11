#!/usr/bin/env python3
"""Train the migrated YOLO11 rice-pest models on AutoDL.

The defaults target the current recommended route:
MEN(P3) + Weighted-P3Fusion.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Avoid OpenMP oversubscription on AutoDL. Users can override this before running the script.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "ultralytics/cfg/models/11/yolo11-men-p3-weighted-p3fusion.yaml"
DEFAULT_DATA = PROJECT_ROOT / "mixdatasets_32classes/data.yaml"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for a reproducible YOLO11 training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model YAML or checkpoint path.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset YAML path.")
    parser.add_argument(
        "--weights",
        default="",
        help="Optional pretrained checkpoint to partially load into a custom YAML model, e.g. yolo11n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="Use -1 for automatic batch-size estimation.")
    parser.add_argument("--device", default="0", help="CUDA device, CPU, or a comma-separated device list.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default=str(PROJECT_ROOT / "runs/train"))
    parser.add_argument("--name", default="yolo11_men_p3_weighted_p3fusion")
    parser.add_argument("--cache", action="store_true", help="Cache images after checking available disk/RAM.")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision training.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing output directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from --weights checkpoint.")
    return parser.parse_args()


def main() -> None:
    """Build the requested model and start training."""
    args = parse_args()
    model_path = Path(args.model)
    data_path = Path(args.data)

    if not data_path.exists() and data_path.suffix.lower() in {".yaml", ".yml"}:
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if model_path.suffix.lower() in {".yaml", ".yml"} and not model_path.exists():
        raise FileNotFoundError(f"Model YAML not found: {model_path}")
    if args.resume and not args.weights:
        raise ValueError("--resume requires --weights pointing to a previous checkpoint, usually last.pt.")

    print(f"model   : {args.model}")
    print(f"data    : {args.data}")
    print(f"device  : {args.device}")
    print(f"epochs  : {args.epochs}, imgsz={args.imgsz}, batch={args.batch}")

    if args.resume:
        model = YOLO(args.weights)
        train_kwargs = {"resume": True}
    else:
        model = YOLO(args.model)
        if args.weights:
            print(f"loading pretrained weights: {args.weights}")
            model.load(args.weights)
        train_kwargs = {
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "patience": args.patience,
            "close_mosaic": args.close_mosaic,
            "seed": args.seed,
            "project": args.project,
            "name": args.name,
            "cache": args.cache,
            "amp": not args.no_amp,
            "exist_ok": args.exist_ok,
            "pretrained": False,
        }

    results = model.train(**train_kwargs)
    save_dir = getattr(results, "save_dir", None) or getattr(model, "trainer", None).save_dir
    print(f"training finished; results saved to: {save_dir}")


if __name__ == "__main__":
    main()

