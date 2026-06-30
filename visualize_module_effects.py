"""
Visualize why YOLO11 improvement modules help.

This script compares multiple YOLO models on the same images and exports:
1. detection-result comparison;
2. feature-map heatmaps from a selected layer;
3. heatmap overlays on the original image;
4. learned WeightedConcat weights for Weighted-P3Fusion-style modules.

Example on the AutoDL server:

python visualize_module_effects.py \
  --source /root/ultralytics-8.3.27/vis_images \
  --out runs/visualize/men_weighted_p3fusion \
  --imgsz 640 \
  --device 0 \
  --model "name=YOLO11n,weights=/root/ultralytics-8.3.27/runs/train/yolo11_mixed_dataset/weights/best.pt,layer=4" \
  --model "name=Weighted-P3Fusion,yaml=/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11-cse-p3-weighted-p3fusion.yaml,weights=/root/ultralytics-8.3.27/runs/train/yolo11_weighted-p3fusion/weights/best.pt,layer=16" \
  --model "name=MEN(P3),yaml=/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11-men-p3.yaml,weights=/root/ultralytics-8.3.27/runs/train/yolo11_men-p3_seed1_mixed_dataset/weights/best.pt,layer=4" \
  --model "name=MEN+Weighted-P3Fusion,yaml=/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11-men-p3-weighted-p3fusion.yaml,weights=/root/ultralytics-8.3.27/runs/train/yolo11_men-p3_weighted-p3fusion_pretrained_mixed_dataset/weights/best.pt,layer=16"

Notes:
- For MEN(P3), layer=4 visualizes the P3/8 backbone output after MEN.
- For Weighted-P3Fusion, layer=16 usually visualizes the P3 branch after the weighted fusion and following C3k2.
- If your YAML layer numbers differ, check the model summary and adjust layer=...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_model_spec(spec: str) -> Dict[str, str]:
    """Parse a model spec like 'name=A,yaml=a.yaml,weights=a.pt,layer=4'."""
    item: Dict[str, str] = {}
    for part in spec.split(","):
        if "=" not in part:
            raise ValueError(f"Invalid --model part '{part}' in '{spec}'. Expected key=value.")
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Invalid --model part '{part}' in '{spec}'. Empty key or value.")
        item[key] = value
    if "name" not in item:
        raise ValueError(f"Model spec requires name=...: {spec}")
    if "weights" not in item:
        raise ValueError(f"Model spec requires weights=...: {spec}")
    return item


def iter_images(source: Path) -> List[Path]:
    """Return image paths from a file or directory."""
    if source.is_file():
        if source.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Source file is not a supported image: {source}")
        return [source]
    if not source.exists():
        raise FileNotFoundError(source)
    images = [p for p in sorted(source.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found under {source}")
    return images


def safe_name(name: str) -> str:
    """Create a filesystem-safe short name."""
    keep = []
    for ch in name:
        keep.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(keep).strip("_") or "model"


def load_yolo(spec: Dict[str, str]) -> YOLO:
    """Load a YOLO model from optional YAML and weights."""
    weights = spec["weights"]
    if "yaml" in spec:
        model = YOLO(spec["yaml"])
        model.load(weights)
        return model
    return YOLO(weights)


def get_layer(model: YOLO, layer_index: int):
    """Resolve a layer index from an Ultralytics YOLO model."""
    layers = getattr(model.model, "model", None)
    if layers is None:
        raise AttributeError("Cannot find model.model layers on the YOLO object.")
    if layer_index < 0:
        layer_index = len(layers) + layer_index
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"Layer index {layer_index} out of range. Model has {len(layers)} layers.")
    return layers[layer_index]


def tensor_to_heatmap(feature: torch.Tensor, out_hw) -> np.ndarray:
    """Convert a feature tensor to a normalized uint8 heatmap."""
    if isinstance(feature, (list, tuple)):
        feature = feature[0]
    if feature.ndim == 3:
        feature = feature.unsqueeze(0)
    if feature.ndim != 4:
        raise ValueError(f"Expected 4D feature tensor, got shape {tuple(feature.shape)}")
    fmap = feature.detach().float().abs().mean(dim=1)[0]
    fmap = fmap.cpu().numpy()
    fmap = fmap - float(fmap.min())
    denom = float(fmap.max()) + 1e-8
    fmap = fmap / denom
    fmap = cv2.resize(fmap, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)
    return np.uint8(np.clip(fmap * 255.0, 0, 255))


def colorize_heatmap(gray_heatmap: np.ndarray) -> np.ndarray:
    """Apply a jet colormap to a uint8 heatmap."""
    return cv2.applyColorMap(gray_heatmap, cv2.COLORMAP_JET)


def overlay_heatmap(image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    """Overlay heatmap on image."""
    return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)


def add_title(image_bgr: np.ndarray, title: str, height: int = 38) -> np.ndarray:
    """Add a simple title bar above an image."""
    h, w = image_bgr.shape[:2]
    canvas = np.full((h + height, w, 3), 255, dtype=np.uint8)
    canvas[height:] = image_bgr
    cv2.rectangle(canvas, (0, 0), (w, height), (245, 248, 252), -1)
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 65, 152), 2, cv2.LINE_AA)
    return canvas


def resize_to_width(image_bgr: np.ndarray, width: int) -> np.ndarray:
    """Resize an image to a fixed width while keeping aspect ratio."""
    h, w = image_bgr.shape[:2]
    if w == width:
        return image_bgr
    new_h = int(round(h * width / w))
    return cv2.resize(image_bgr, (width, new_h), interpolation=cv2.INTER_AREA)


def pad_to_shape(image_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Pad an image to the requested shape."""
    h, w = image_bgr.shape[:2]
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[:h, :w] = image_bgr
    return canvas


def make_grid(rows: List[List[np.ndarray]], gap: int = 8) -> np.ndarray:
    """Create a white-grid montage from BGR images."""
    row_imgs = []
    max_width = 0
    for row in rows:
        max_h = max(img.shape[0] for img in row)
        total_w = sum(img.shape[1] for img in row) + gap * (len(row) - 1)
        canvas = np.full((max_h, total_w, 3), 255, dtype=np.uint8)
        x = 0
        for img in row:
            padded = pad_to_shape(img, max_h, img.shape[1])
            canvas[:, x : x + img.shape[1]] = padded
            x += img.shape[1] + gap
        row_imgs.append(canvas)
        max_width = max(max_width, total_w)
    total_h = sum(img.shape[0] for img in row_imgs) + gap * (len(row_imgs) - 1)
    montage = np.full((total_h, max_width, 3), 255, dtype=np.uint8)
    y = 0
    for img in row_imgs:
        montage[y : y + img.shape[0], : img.shape[1]] = img
        y += img.shape[0] + gap
    return montage


def extract_weighted_concat_weights(model: YOLO) -> List[Dict[str, object]]:
    """Extract normalized weights from all WeightedConcat modules in a model."""
    data: List[Dict[str, object]] = []
    for idx, module in enumerate(getattr(model.model, "model", [])):
        if module.__class__.__name__ != "WeightedConcat":
            continue
        raw = module.w.detach().float().cpu()
        relu = torch.relu(raw)
        norm = relu / (relu.sum() + float(module.eps))
        data.append(
            {
                "layer": idx,
                "raw": [round(float(v), 6) for v in raw],
                "normalized": [round(float(v), 6) for v in norm],
            }
        )
    return data


def run_one_model(
    model: YOLO,
    image_path: Path,
    layer_index: int,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
):
    """Run prediction with a hook and return detection, feature heatmap, and overlay images."""
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        captured["feature"] = output[0] if isinstance(output, (list, tuple)) else output

    handle = get_layer(model, layer_index).register_forward_hook(hook_fn)
    try:
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
    finally:
        handle.remove()

    if not results:
        raise RuntimeError(f"No prediction result for {image_path}")
    result = results[0]
    original_bgr = result.orig_img.copy()
    det_bgr = result.plot()
    if "feature" not in captured:
        raise RuntimeError(f"Layer {layer_index} did not capture any feature for {image_path}")
    heat_gray = tensor_to_heatmap(captured["feature"], original_bgr.shape[:2])
    heat_bgr = colorize_heatmap(heat_gray)
    overlay_bgr = overlay_heatmap(original_bgr, heat_bgr)
    return det_bgr, heat_bgr, overlay_bgr


def write_json(path: Path, payload) -> None:
    """Write JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate YOLO module-effect visualizations.")
    parser.add_argument("--source", required=True, type=Path, help="Image file or directory.")
    parser.add_argument("--out", default=Path("runs/visualize/module_effects"), type=Path, help="Output directory.")
    parser.add_argument("--model", action="append", required=True, help="Model spec: name=...,weights=...,yaml=...,layer=...")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.7, type=float)
    parser.add_argument("--device", default="0", help="CUDA device id or cpu.")
    parser.add_argument("--max-images", default=12, type=int, help="Maximum number of images to visualize.")
    parser.add_argument("--tile-width", default=300, type=int, help="Width of each model tile in the montage.")
    parser.add_argument("--save-single", action="store_true", help="Also save each individual model image.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.source)[: args.max_images]
    specs = [parse_model_spec(s) for s in args.model]

    loaded = []
    all_weights: Dict[str, object] = {}
    for spec in specs:
        model = load_yolo(spec)
        model_name = spec["name"]
        layer_index = int(spec.get("layer", "4"))
        loaded.append((model_name, model, layer_index))
        all_weights[model_name] = extract_weighted_concat_weights(model)
    write_json(args.out / "weighted_concat_weights.json", all_weights)

    for image_path in image_paths:
        stem = image_path.stem
        per_image_dir = args.out / stem
        if args.save_single:
            per_image_dir.mkdir(parents=True, exist_ok=True)

        detection_row: List[np.ndarray] = []
        feature_row: List[np.ndarray] = []
        overlay_row: List[np.ndarray] = []

        for model_name, model, layer_index in loaded:
            print(f"[{stem}] {model_name}: layer={layer_index}")
            det, feature, overlay = run_one_model(model, image_path, layer_index, args.imgsz, args.conf, args.iou, args.device)
            det = resize_to_width(det, args.tile_width)
            feature = resize_to_width(feature, args.tile_width)
            overlay = resize_to_width(overlay, args.tile_width)

            detection_row.append(add_title(det, f"{model_name} | Detection"))
            feature_row.append(add_title(feature, f"{model_name} | Feature map"))
            overlay_row.append(add_title(overlay, f"{model_name} | Heatmap overlay"))

            if args.save_single:
                name = safe_name(model_name)
                cv2.imwrite(str(per_image_dir / f"{name}_detection.jpg"), det)
                cv2.imwrite(str(per_image_dir / f"{name}_feature_map.jpg"), feature)
                cv2.imwrite(str(per_image_dir / f"{name}_heatmap_overlay.jpg"), overlay)

        montage = make_grid([detection_row, feature_row, overlay_row])
        cv2.imwrite(str(args.out / f"{stem}_module_effects.jpg"), montage)

    print(f"Saved visualizations to: {args.out.resolve()}")
    print(f"Saved WeightedConcat weights to: {(args.out / 'weighted_concat_weights.json').resolve()}")


if __name__ == "__main__":
    main()
