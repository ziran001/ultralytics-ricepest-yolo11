"""
Find small-object cases missed by a baseline YOLO model but detected by an improved model.

The script compares two trained YOLO weights against YOLO-format labels and saves
case folders that are useful for papers or PPTs:

- original.jpg
- gt.jpg
- baseline.jpg
- improved.jpg
- comparison.jpg

Example:
python find_baseline_missed_cases.py \
  --data /root/ultralytics-8.3.27/mixed_dataset/data.yaml \
  --split val \
  --baseline /root/ultralytics-8.3.27/runs/train/yolo11_mixed_dataset/weights/best.pt \
  --improved /root/ultralytics-8.3.27/runs/train/yolo11_p2head_pretrained_mixed_dataset/weights/best.pt \
  --out runs/visualize/missed_cases_p2head \
  --classes 1,2,22 \
  --small-area 0.01 \
  --conf 0.25 \
  --match-iou 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_ALIASES = {
    "Chilo suppressalis": "C. suppressalis",
    "Cnaphalocrocis medinalis": "C. medinalis",
    "Rice plant hopper": "RPH",
    "Ostrinia furnacalis": "O. furnacalis",
    "Scirpophaga incertulas": "S. incertulas",
    "Spodoptera frugiperda": "S. frugiperda",
    "Spodoptera spp.": "Spodoptera",
    "Sesamia inferens": "S. inferens",
    "Anomala corpulenta": "A. corpulenta",
    "Holotrichia diomphalia": "H. diomphalia",
    "Hydrochara affinis": "H. affinis",
    "Sirthenea flavipes": "S. flavipes",
    "Naranga aenescens": "N. aenescens",
    "copper-green chafer": "chafer",
    "Anomala exoleta Fald": "A. exoleta",
    "Plutella xylostella": "P. xylostella",
    "Agrotis segetum": "A. segetum",
    "Axylia putris": "A. putris",
    "Athetis spp.": "Athetis",
    "Helicoverpa armigera": "H. armigera",
}
CLASS_COLORS_BGR = [
    (91, 64, 255),
    (60, 210, 80),
    (0, 205, 255),
    (180, 120, 255),
    (64, 128, 255),
    (80, 180, 220),
    (180, 180, 60),
    (220, 120, 60),
    (128, 180, 80),
    (220, 90, 170),
    (80, 200, 200),
    (120, 120, 255),
    (150, 210, 100),
    (255, 170, 60),
    (190, 140, 80),
    (120, 80, 220),
    (80, 160, 255),
    (100, 220, 160),
    (180, 90, 120),
    (220, 180, 90),
    (90, 180, 230),
    (150, 100, 220),
    (220, 60, 220),
]
CLASS_COLOR_OVERRIDES_BGR = {
    "Chilo suppressalis": (60, 210, 80),
    "Cnaphalocrocis medinalis": (0, 205, 255),
    "Rice plant hopper": (220, 60, 220),
}


def safe_name(name: str) -> str:
    """Create a filesystem-safe short name."""
    keep = []
    for ch in name:
        keep.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(keep).strip("_") or "case"


def class_name(names, class_id: int) -> str:
    """Return a class name from dict/list names."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def short_label(name: str, full: bool = False) -> str:
    """Return compact class label for drawing."""
    return name if full else CLASS_ALIASES.get(name, name)


def class_color_bgr(name: str, class_id: int) -> Tuple[int, int, int]:
    """Return a stable BGR color for a class."""
    if name in CLASS_COLOR_OVERRIDES_BGR:
        return CLASS_COLOR_OVERRIDES_BGR[name]
    return CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]


def readable_text_color(bg_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Choose black or white text according to background brightness."""
    b, g, r = bg_bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (20, 20, 20) if luminance > 155 else (255, 255, 255)


def resolve_dataset_path(data_yaml: Path, value: str) -> Path:
    """Resolve a train/val path from Ultralytics data.yaml."""
    p = Path(value)
    if p.is_absolute():
        return p
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root / p


def load_data(data_yaml: Path, split: str) -> Tuple[Path, Path, List[str]]:
    """Load image/label paths and class names from data.yaml."""
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if split not in data:
        raise KeyError(f"{split!r} not found in {data_yaml}")
    image_dir = resolve_dataset_path(data_yaml, data[split])
    label_dir = Path(str(image_dir).replace(f"{split}/images", f"{split}/labels"))
    if label_dir == image_dir:
        label_dir = image_dir.parent.parent / split / "labels"
    names = data.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    return image_dir, label_dir, names


def iter_images(image_dir: Path, max_images: int) -> List[Path]:
    """Return image list."""
    images = [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    return images[:max_images] if max_images > 0 else images


def read_yolo_labels(label_path: Path, image_shape: Tuple[int, int]) -> List[Dict[str, object]]:
    """Read YOLO txt labels into xyxy boxes."""
    h, w = image_shape
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, bw, bh = map(float, parts[1:5])
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h
        labels.append(
            {
                "cls": cls_id,
                "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                "area_ratio": float(bw * bh),
            }
        )
    return labels


def predictions_from_result(result) -> List[Dict[str, object]]:
    """Extract model predictions from an Ultralytics result."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy().astype(float)
    preds = []
    for box, c, score in zip(xyxy, cls, conf):
        preds.append({"cls": int(c), "xyxy": box, "conf": float(score)})
    return preds


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two xyxy boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / (area_a + area_b - inter + 1e-9)


def best_match(gt: Dict[str, object], preds: List[Dict[str, object]], match_iou: float) -> Optional[Dict[str, object]]:
    """Return best same-class prediction matched to one GT."""
    best = None
    best_iou = 0.0
    for pred in preds:
        if int(pred["cls"]) != int(gt["cls"]):
            continue
        iou = box_iou(gt["xyxy"], pred["xyxy"])
        if iou > best_iou:
            best_iou = iou
            best = pred
    if best is not None and best_iou >= match_iou:
        item = dict(best)
        item["iou"] = best_iou
        return item
    return None


def draw_box_with_label(
    image: np.ndarray,
    box: np.ndarray,
    label: str,
    color: Tuple[int, int, int],
    show_label: bool = True,
    thickness: int = 2,
) -> None:
    """Draw one colored box with compact label."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if not show_label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.34, min(0.5, min(h, w) / 1850))
    text_size, baseline = cv2.getTextSize(label, font, scale, 1)
    tw, th = text_size
    label_w = tw + 6
    label_h = th + baseline + 6
    lx1 = min(max(0, x1), max(0, w - label_w))
    ly1 = y1 - label_h - 1
    if ly1 < 0:
        ly1 = min(h - label_h, y1 + 1)
    lx2, ly2 = lx1 + label_w, ly1 + label_h
    cv2.rectangle(image, (lx1, ly1), (lx2, ly2), color, -1)
    cv2.putText(image, label, (lx1 + 3, ly2 - baseline - 3), font, scale, readable_text_color(color), 1, cv2.LINE_AA)


def draw_gt(image: np.ndarray, labels: List[Dict[str, object]], names: List[str], target_idx: int, full_label: bool) -> np.ndarray:
    """Draw ground truth boxes, highlighting the target missed case in red."""
    out = image.copy()
    for idx, gt in enumerate(labels):
        cls_id = int(gt["cls"])
        name = class_name(names, cls_id)
        color = (0, 0, 255) if idx == target_idx else class_color_bgr(name, cls_id)
        label = short_label(name, full_label)
        draw_box_with_label(out, gt["xyxy"], label, color, show_label=True, thickness=3 if idx == target_idx else 2)
    return out


def draw_predictions(
    image: np.ndarray,
    preds: List[Dict[str, object]],
    names: List[str],
    focus_box: np.ndarray,
    title_miss: bool,
    full_label: bool,
) -> np.ndarray:
    """Draw predictions and mark the GT focus box in red if missed."""
    out = image.copy()
    for pred in preds:
        cls_id = int(pred["cls"])
        name = class_name(names, cls_id)
        color = class_color_bgr(name, cls_id)
        label = f"{short_label(name, full_label)} {float(pred['conf']):.2f}"
        draw_box_with_label(out, pred["xyxy"], label, color, show_label=True, thickness=2)
    if title_miss:
        cv2.rectangle(
            out,
            (int(focus_box[0]), int(focus_box[1])),
            (int(focus_box[2]), int(focus_box[3])),
            (0, 0, 255),
            4,
        )
    return out


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize image to fixed width."""
    h, w = image.shape[:2]
    if w == width:
        return image
    return cv2.resize(image, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def add_caption(image: np.ndarray, caption: str, height: int = 44) -> np.ndarray:
    """Add bottom caption for comparison panels."""
    h, w = image.shape[:2]
    canvas = np.full((h + height, w, 3), 255, dtype=np.uint8)
    canvas[:h] = image
    scale = min(0.72, max(0.38, w / max(1, len(caption)) / 13.0))
    thickness = 1 if scale < 0.6 else 2
    text_size, _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(4, (w - text_size[0]) // 2)
    y = h + 28
    cv2.putText(canvas, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), thickness, cv2.LINE_AA)
    return canvas


def make_comparison(gt_img: np.ndarray, baseline_img: np.ndarray, improved_img: np.ndarray, width: int) -> np.ndarray:
    """Create a three-panel comparison image."""
    panels = [
        add_caption(resize_to_width(gt_img, width), "(a) GT"),
        add_caption(resize_to_width(baseline_img, width), "(b) Baseline missed"),
        add_caption(resize_to_width(improved_img, width), "(c) Improved detected"),
    ]
    max_h = max(p.shape[0] for p in panels)
    gap = 12
    total_w = sum(p.shape[1] for p in panels) + gap * 2
    canvas = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for p in panels:
        canvas[: p.shape[0], x : x + p.shape[1]] = p
        x += p.shape[1] + gap
    return canvas


def crop_focus(image: np.ndarray, box: np.ndarray, ratio: float) -> np.ndarray:
    """Crop around focus box with context."""
    if ratio <= 0:
        return image
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    pad = max(bw, bh) * ratio
    cx1 = max(0, int(round(x1 - pad)))
    cy1 = max(0, int(round(y1 - pad)))
    cx2 = min(w, int(round(x2 + pad)))
    cy2 = min(h, int(round(y2 + pad)))
    return image[cy1:cy2, cx1:cx2]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    """Write candidate CSV."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_case(
    out_dir: Path,
    case_id: int,
    image_path: Path,
    image: np.ndarray,
    labels: List[Dict[str, object]],
    target_idx: int,
    baseline_preds: List[Dict[str, object]],
    improved_preds: List[Dict[str, object]],
    names: List[str],
    full_label: bool,
    panel_width: int,
    crop_ratio: float,
) -> Path:
    """Save one candidate case folder."""
    gt = labels[target_idx]
    cls_id = int(gt["cls"])
    cls_name = safe_name(short_label(class_name(names, cls_id), full_label))
    case_dir = out_dir / f"case_{case_id:03d}_{cls_name}_{image_path.stem}"
    case_dir.mkdir(parents=True, exist_ok=True)

    gt_img = draw_gt(image, labels, names, target_idx, full_label)
    baseline_img = draw_predictions(image, baseline_preds, names, gt["xyxy"], title_miss=True, full_label=full_label)
    improved_img = draw_predictions(image, improved_preds, names, gt["xyxy"], title_miss=False, full_label=full_label)

    cv2.imwrite(str(case_dir / "original.jpg"), image)
    cv2.imwrite(str(case_dir / "gt.jpg"), gt_img)
    cv2.imwrite(str(case_dir / "baseline_missed.jpg"), baseline_img)
    cv2.imwrite(str(case_dir / "improved_detected.jpg"), improved_img)
    cv2.imwrite(str(case_dir / "comparison.jpg"), make_comparison(gt_img, baseline_img, improved_img, panel_width))

    if crop_ratio > 0:
        cv2.imwrite(str(case_dir / "crop_gt.jpg"), crop_focus(gt_img, gt["xyxy"], crop_ratio))
        cv2.imwrite(str(case_dir / "crop_baseline_missed.jpg"), crop_focus(baseline_img, gt["xyxy"], crop_ratio))
        cv2.imwrite(str(case_dir / "crop_improved_detected.jpg"), crop_focus(improved_img, gt["xyxy"], crop_ratio))

    shutil.copy2(image_path, case_dir / image_path.name)
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Find baseline-missed small-object cases detected by an improved model.")
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics data.yaml path.")
    parser.add_argument("--split", default="val", choices=["train", "val"], help="Dataset split to scan.")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline best.pt.")
    parser.add_argument("--improved", required=True, type=Path, help="Improved best.pt.")
    parser.add_argument("--out", default=Path("runs/visualize/missed_cases"), type=Path, help="Output directory.")
    parser.add_argument("--classes", default="1,2,22", help="Target class ids, comma separated. Default: 1,2,22.")
    parser.add_argument("--small-area", default=0.01, type=float, help="GT area ratio threshold for small objects.")
    parser.add_argument("--conf", default=0.25, type=float, help="Prediction confidence threshold.")
    parser.add_argument("--pred-iou", default=0.7, type=float, help="NMS IoU for YOLO prediction.")
    parser.add_argument("--match-iou", default=0.5, type=float, help="IoU threshold for GT-prediction match.")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", default=0, type=int, help="0 means scan all images.")
    parser.add_argument("--max-cases", default=30, type=int, help="Maximum saved cases.")
    parser.add_argument("--panel-width", default=640, type=int, help="Comparison panel width.")
    parser.add_argument("--crop-ratio", default=5.0, type=float, help="Context crop ratio around the missed object; 0 disables.")
    parser.add_argument("--full-label", action="store_true", help="Use full class names instead of compact labels.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    target_classes = {int(x) for x in args.classes.split(",") if x.strip()}
    image_dir, label_dir, names = load_data(args.data, args.split)
    image_paths = iter_images(image_dir, args.max_images)

    baseline = YOLO(str(args.baseline))
    improved = YOLO(str(args.improved))
    rows: List[Dict[str, object]] = []
    saved = 0

    print(f"Scanning {len(image_paths)} images from {image_dir}")
    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_yolo_labels(label_path, (h, w))
        labels = [gt for gt in labels if int(gt["cls"]) in target_classes and float(gt["area_ratio"]) <= args.small_area]
        if not labels:
            continue

        baseline_result = baseline.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, iou=args.pred_iou, device=args.device, verbose=False)[0]
        improved_result = improved.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, iou=args.pred_iou, device=args.device, verbose=False)[0]
        baseline_preds = predictions_from_result(baseline_result)
        improved_preds = predictions_from_result(improved_result)

        for target_idx, gt in enumerate(labels):
            baseline_hit = best_match(gt, baseline_preds, args.match_iou)
            improved_hit = best_match(gt, improved_preds, args.match_iou)
            if baseline_hit is not None or improved_hit is None:
                continue

            saved += 1
            case_dir = save_case(
                args.out,
                saved,
                image_path,
                image,
                labels,
                target_idx,
                baseline_preds,
                improved_preds,
                names,
                args.full_label,
                args.panel_width,
                args.crop_ratio,
            )
            cls_id = int(gt["cls"])
            rows.append(
                {
                    "case_id": saved,
                    "image": str(image_path),
                    "case_dir": str(case_dir),
                    "class_id": cls_id,
                    "class_name": class_name(names, cls_id),
                    "gt_area_ratio": round(float(gt["area_ratio"]), 6),
                    "improved_conf": round(float(improved_hit["conf"]), 4),
                    "improved_iou": round(float(improved_hit["iou"]), 4),
                    "gt_xyxy": json.dumps([round(float(v), 2) for v in gt["xyxy"]]),
                }
            )
            print(f"[{saved:03d}] {class_name(names, cls_id)} missed by baseline, detected by improved: {case_dir}")
            if saved >= args.max_cases:
                write_csv(args.out / "candidates.csv", rows)
                print(f"Saved {saved} cases to {args.out}")
                return

        if idx % 100 == 0:
            print(f"Scanned {idx}/{len(image_paths)} images, saved {saved} cases")

    write_csv(args.out / "candidates.csv", rows)
    print(f"Done. Saved {saved} cases to {args.out}")


if __name__ == "__main__":
    main()
