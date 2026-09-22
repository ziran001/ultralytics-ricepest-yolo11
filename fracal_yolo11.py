"""FRACAL (CVPR 2025) post-calibration experiment for YOLO11 detection.

Paper: https://openaccess.thecvf.com/content/CVPR2025/html/Alexandridis_Fractal_Calibration_for_Long-tailed_Object_Detection_CVPR_2025_paper.html
This script implements the paper's box-counting statistic and sigmoid-detector
Equation (11), applied to YOLO11 class scores before confidence filtering/NMS.
It does not train or alter the checkpoint.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def dataset_config(data_file):
    """Resolve the data root, train source, and ordered class names."""
    data_file = Path(data_file).expanduser().resolve()
    with data_file.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    root = Path(data.get("path", data_file.parent))
    if not root.is_absolute():
        root = data_file.parent / root
    root = root.resolve()
    names = data["names"]
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if not names or any(name is None for name in names):
        raise ValueError("Dataset names must contain contiguous class IDs starting at 0")
    train = data["train"]
    sources = train if isinstance(train, list) else [train]
    return root, sources, [str(name) for name in names]


def training_images(root, sources):
    """Yield every training entry, retaining repeated paths in weighted manifests."""
    for source in sources:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = root / source_path
        source_path = source_path.resolve()
        if source_path.suffix.lower() == ".txt":
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            with source_path.open(encoding="utf-8") as stream:
                for line in stream:
                    entry = line.strip()
                    if not entry:
                        continue
                    image = Path(entry)
                    if not image.is_absolute():
                        image = root / image
                    yield image.resolve()
        elif source_path.is_dir():
            yield from (p for p in sorted(source_path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES)
        elif source_path.suffix.lower() in IMAGE_SUFFIXES and source_path.is_file():
            yield source_path
        else:
            raise FileNotFoundError(f"Unsupported or missing train source: {source_path}")


def label_path(image):
    """Use the same images/ -> labels/ path convention as Ultralytics."""
    parts = list(image.parts)
    positions = [i for i, part in enumerate(parts) if part == "images"]
    if not positions:
        raise ValueError(f"Image path has no 'images' directory: {image}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def box_dimension(centers, max_grid=64):
    """Fit log(occupied cells) against log(grid size), as in FRACAL."""
    n = len(centers)
    if n < 4:
        return 1.0
    max_grid = min(max_grid, math.isqrt(n))
    grids = np.arange(1, max_grid + 1, dtype=np.int64)
    xy = np.clip(np.asarray(centers, dtype=np.float64), 0.0, np.nextafter(1.0, 0.0))
    occupied = []
    for grid in grids:
        cells = np.floor(xy * grid).astype(np.int64)
        occupied.append(np.unique(cells[:, 0] + grid * cells[:, 1]).size)
    slope = np.polyfit(np.log(grids), np.log(occupied), 1)[0]
    return float(max(1.0, slope))


def collect_statistics(data_file, max_grid=64):
    """Count actual train exposures and compute classwise spatial dimensions."""
    root, sources, names = dataset_config(data_file)
    counts = np.zeros(len(names), dtype=np.int64)
    centers = [[] for _ in names]
    image_count = 0
    for image in training_images(root, sources):
        if not image.is_file():
            raise FileNotFoundError(f"Missing training image: {image}")
        image_count += 1
        labels = label_path(image)
        if not labels.exists():  # YOLO permits background images without a label file.
            continue
        with labels.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(f"Expected YOLO detection label with 5 fields: {labels}:{line_number}")
                cls_float = float(fields[0])
                cls = int(cls_float)
                x, y = map(float, fields[1:3])
                if cls_float != cls or not 0 <= cls < len(names) or not 0 <= x <= 1 or not 0 <= y <= 1:
                    raise ValueError(f"Invalid class or center: {labels}:{line_number}")
                counts[cls] += 1
                centers[cls].append((x, y))
    if not image_count or np.any(counts == 0):
        raise ValueError(f"Train manifest is empty or some classes have no labels: {counts.tolist()}")
    return {
        "method": "FRACAL box-counting, weighted train entries",
        "data": str(Path(data_file).expanduser().resolve()),
        "train_sources": [str(source) for source in sources],
        "num_train_entries": image_count,
        "names": names,
        "instance_counts": counts.tolist(),
        "fractal_dimensions": [box_dimension(items, max_grid=max_grid) for items in centers],
        "max_grid": max_grid,
    }


def calibration_shift(stats, beta=10.0, fractal_lambda=2.0):
    """Class and spatial logit offsets from FRACAL Eq. (6) and Eq. (11)."""
    if beta <= 1 or fractal_lambda < 0:
        raise ValueError("Require beta > 1 and fractal_lambda >= 0")
    counts = np.asarray(stats["instance_counts"], dtype=np.float64)
    dims = np.asarray(stats["fractal_dimensions"], dtype=np.float64)
    if counts.ndim != 1 or len(counts) != len(dims) or np.any(counts <= 0) or np.any(dims <= 0):
        raise ValueError("Invalid or mismatched frequency/fractal statistics")
    nclass = len(counts)
    frequency = np.log(counts.sum() / (nclass * counts)) / np.log(beta)
    spatial_weights = dims**fractal_lambda
    spatial = -np.log(spatial_weights / spatial_weights.sum()) / np.log(beta) + math.log(1 / nclass, beta)
    return frequency + spatial


class FracalCalibrator:
    """Apply the paper's binary-classifier rule: sigmoid(z + offset) * sigmoid(z)."""

    def __init__(self, stats, beta=10.0, fractal_lambda=2.0):
        self.names = stats["names"]
        self.offsets = calibration_shift(stats, beta, fractal_lambda)
        self._cached_offsets = {}

    def __call__(self, prediction):
        import torch

        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0]
        nc = len(self.names)
        if prediction.ndim != 3 or prediction.shape[1] != nc + 4:
            raise ValueError(f"Expected YOLO detection output [batch, {nc + 4}, boxes]; got {tuple(prediction.shape)}")
        key = (prediction.device, prediction.dtype)
        if key not in self._cached_offsets:
            self._cached_offsets[key] = torch.as_tensor(self.offsets, device=prediction.device, dtype=torch.float32)[
                None, :, None
            ]
        probabilities = prediction[:, 4 : 4 + nc, :].float().clamp(1e-6, 1 - 1e-6)
        calibrated = probabilities * torch.sigmoid(torch.logit(probabilities) + self._cached_offsets[key])
        output = prediction.clone()
        output[:, 4 : 4 + nc, :] = calibrated.to(prediction.dtype)
        return output


def run_validation(args):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator

    with Path(args.stats).open(encoding="utf-8") as stream:
        stats = json.load(stream)
    _, _, dataset_names = dataset_config(args.data)
    if stats["names"] != dataset_names:
        raise ValueError("Statistics class names/order differ from the validation data.yaml")
    model = YOLO(args.weights)
    model_names = [model.names[i] for i in range(len(model.names))]
    if model_names != dataset_names:
        raise ValueError("Checkpoint class names/order differ from the validation data.yaml")

    calibrator = FracalCalibrator(stats, args.beta, args.fractal_lambda)

    class FracalValidator(DetectionValidator):
        def postprocess(self, preds):
            return super().postprocess(calibrator(preds))

    common = dict(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        exist_ok=False,
        plots=args.plots,
    )
    def metrics_summary(metrics):
        rows = {}
        for index, cls in enumerate(metrics.box.ap_class_index):
            p, r, ap50, ap = metrics.class_result(index)
            rows[dataset_names[int(cls)]] = {
                "P": float(p), "R": float(r), "mAP50": float(ap50), "mAP50-95": float(ap)
            }
        return {
            "P": float(metrics.box.mp),
            "R": float(metrics.box.mr),
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "per_class": rows,
        }

    summary = {"weights": str(Path(args.weights).resolve()), "stats": str(Path(args.stats).resolve())}
    if args.baseline:
        baseline = model.val(name=f"{args.name}_baseline", **common)
        summary["baseline"] = metrics_summary(baseline)
    fracal = model.val(validator=FracalValidator, name=f"{args.name}_fracal", **common)
    summary["fracal"] = metrics_summary(fracal)
    summary["beta"] = args.beta
    summary["fractal_lambda"] = args.fractal_lambda
    summary_path = Path(fracal.save_dir) / "fracal_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved comparison to {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    stats_parser = subparsers.add_parser("stats", help="Compute statistics from the actual training manifest")
    stats_parser.add_argument("--data", required=True, help="32-class data.yaml")
    stats_parser.add_argument("--output", required=True, help="Output JSON file outside the source repository")
    stats_parser.add_argument("--max-grid", type=int, default=64)

    val_parser = subparsers.add_parser("val", help="Validate the same checkpoint with FRACAL")
    val_parser.add_argument("--data", required=True)
    val_parser.add_argument("--weights", required=True, help="YOLO11n baseline best.pt")
    val_parser.add_argument("--stats", required=True, help="JSON generated by the stats command")
    val_parser.add_argument("--project", default="runs/val")
    val_parser.add_argument("--name", default="yolo11n_mixdatasets32_fracal")
    val_parser.add_argument("--imgsz", type=int, default=1024)
    val_parser.add_argument("--batch", type=int, default=32)
    val_parser.add_argument("--device", default="0")
    val_parser.add_argument("--workers", type=int, default=8)
    val_parser.add_argument("--conf", type=float, default=0.001)
    val_parser.add_argument("--iou", type=float, default=0.7)
    val_parser.add_argument("--max-det", type=int, default=300)
    val_parser.add_argument("--beta", type=float, default=10.0)
    val_parser.add_argument("--fractal-lambda", type=float, default=2.0)
    val_parser.add_argument("--baseline", action="store_true", help="Also rerun uncalibrated validation")
    val_parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    if args.command == "stats":
        if args.max_grid < 2:
            parser.error("--max-grid must be at least 2")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        stats = collect_statistics(args.data, args.max_grid)
        output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote statistics for {len(stats['names'])} classes to {output}")
    else:
        run_validation(args)


if __name__ == "__main__":
    main()
