"""
Visualize why YOLO11 improvement modules help.

This script compares multiple YOLO models on the same images and exports:
1. a detection-result comparison figure;
2. a feature-map and heatmap-overlay comparison figure;
3. learned WeightedConcat weights for Weighted-P3Fusion-style modules.

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

Main outputs:
- *_detection_comparison.jpg: (a) original image, followed by (b), (c), ... model detections.
- *_feature_heatmap_comparison.jpg: first row feature maps, second row heatmap overlays.

Optional miss/false-positive marker JSON format:
{
  "image_stem": {
    "Model name": {
      "miss": [[x1, y1, x2, y2]],
      "false_positive": [[x1, y1, x2, y2]]
    }
  }
}

Notes:
- For MEN(P3), layer=4 visualizes the P3/8 backbone output after MEN.
- For Weighted-P3Fusion, layer=16 usually visualizes the P3 branch after the weighted fusion and following C3k2.
- If your YAML layer numbers differ, check the model summary and adjust layer=...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

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


def class_name(names, class_id: int) -> str:
    """Return a class name from Ultralytics names containers."""
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def short_label(name: str, style: str = "short") -> str:
    """Shorten long pest species names for compact paper figures."""
    if style == "full":
        return name
    return CLASS_ALIASES.get(name, name)


def class_color_bgr(name: str, class_id: int) -> tuple:
    """Return a stable BGR color for a class."""
    if name in CLASS_COLOR_OVERRIDES_BGR:
        return CLASS_COLOR_OVERRIDES_BGR[name]
    return CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]


def readable_text_color(bg_bgr: tuple) -> tuple:
    """Choose black or white text according to background brightness."""
    b, g, r = bg_bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (20, 20, 20) if luminance > 155 else (255, 255, 255)


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


def get_model_names(model: YOLO):
    """Return class names from a YOLO wrapper or its inner model."""
    names = getattr(model, "names", None)
    if names:
        return names
    inner = getattr(model, "model", None)
    return getattr(inner, "names", {})


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


def split_caption(caption: str, width: int, scale: float, thickness: int) -> List[str]:
    """Split a long caption into one or two centered lines."""
    text_size, _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if text_size[0] <= width - 10:
        return [caption]
    if " " in caption:
        first, second = caption.split(" ", 1)
        return [first, second]
    for token in ["+", "-", "_"]:
        if token in caption:
            left, right = caption.split(token, 1)
            return [left + token, right]
    mid = max(1, len(caption) // 2)
    return [caption[:mid], caption[mid:]]


def add_caption(image_bgr: np.ndarray, caption: str, height: int = 54) -> np.ndarray:
    """Add a bottom caption similar to paper comparison figures."""
    h, w = image_bgr.shape[:2]
    canvas = np.full((h + height, w, 3), 255, dtype=np.uint8)
    canvas[:h] = image_bgr
    cv2.rectangle(canvas, (0, h), (w, h + height), (255, 255, 255), -1)
    scale = min(0.64, max(0.36, w / max(1, len(caption)) / 13.5))
    thickness = 1 if scale < 0.6 else 2
    lines = split_caption(caption, w, scale, thickness)
    y0 = h + 21 if len(lines) == 1 else h + 18
    for line_id, line in enumerate(lines[:2]):
        while cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > w - 10 and scale > 0.28:
            scale -= 0.02
            thickness = 1
        text_size, _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = max(5, (w - text_size[0]) // 2)
        y = y0 + line_id * 22
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), thickness, cv2.LINE_AA)
    return canvas


def draw_error_markers(image_bgr: np.ndarray, markers: Dict[str, object]) -> np.ndarray:
    """Draw red rectangles for misses and yellow ellipses for false positives."""
    if not markers:
        return image_bgr
    marked = image_bgr.copy()
    h, w = marked.shape[:2]
    line_width = max(2, int(round(min(h, w) / 260)))

    for box in markers.get("miss", []):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 0, 255), line_width)

    for box in markers.get("false_positive", []):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(4, abs(x2 - x1) // 2), max(4, abs(y2 - y1) // 2))
        cv2.ellipse(marked, center, axes, 0, 0, 360, (0, 255, 255), line_width)

    return marked


def draw_box_with_label(
    image_bgr: np.ndarray,
    xyxy: List[float],
    label: str,
    color=(60, 210, 80),
    label_bg=None,
    label_fg=None,
    show_label: bool = True,
) -> None:
    """Draw one detection box with a compact, clipped label."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return

    line_width = max(2, int(round(min(h, w) / 420)))
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, line_width)
    if not show_label or not label:
        return

    label_bg = color if label_bg is None else label_bg
    label_fg = readable_text_color(label_bg) if label_fg is None else label_fg
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.32, min(0.48, min(h, w) / 1850))
    thickness = 1
    max_label_w = max(40, w - 8)
    while cv2.getTextSize(label, font, scale, thickness)[0][0] > max_label_w and scale > 0.25:
        scale -= 0.02
    text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
    tw, th = text_size
    pad_x, pad_y = 3, 3
    label_w = tw + pad_x * 2
    label_h = th + pad_y * 2 + baseline
    lx1 = min(max(0, x1), max(0, w - label_w))
    ly1 = y1 - label_h - 1
    if ly1 < 0:
        ly1 = min(h - label_h, y1 + 1)
    ly2 = ly1 + label_h
    lx2 = lx1 + label_w
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), label_bg, -1)
    cv2.addWeighted(overlay, 0.72, image_bgr, 0.28, 0, image_bgr)
    cv2.rectangle(image_bgr, (lx1, ly1), (lx2, ly2), label_bg, 1)
    cv2.putText(image_bgr, label, (lx1 + pad_x, ly2 - pad_y - baseline), font, scale, label_fg, thickness, cv2.LINE_AA)


def draw_predictions(result, label_style: str, show_labels: bool = True) -> np.ndarray:
    """Draw prediction boxes with compact labels instead of Ultralytics default plot()."""
    image = result.orig_img.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return image
    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    order = np.argsort(-areas)
    for i in order:
        full_name = class_name(result.names, int(cls[i]))
        name = short_label(full_name, label_style)
        label = f"{name} {conf[i]:.2f}"
        color = class_color_bgr(full_name, int(cls[i]))
        draw_box_with_label(
            image,
            xyxy[i].tolist(),
            label,
            color=color,
            show_label=show_labels,
        )
    return image


def find_label_path(image_path: Path) -> Optional[Path]:
    """Find a YOLO txt label stored beside the image or in a sibling labels directory."""
    same_dir = image_path.with_suffix(".txt")
    if same_dir.exists():
        return same_dir
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        label_parts = parts[:]
        label_parts[idx] = "labels"
        label_path = Path(*label_parts).with_suffix(".txt")
        if label_path.exists():
            return label_path
    return None


def draw_ground_truth(image_bgr: np.ndarray, image_path: Path, names, label_style: str, show_labels: bool = True) -> np.ndarray:
    """Draw YOLO-format ground-truth labels for the original-image panel."""
    label_path = find_label_path(image_path)
    if label_path is None:
        return image_bgr
    image = image_bgr.copy()
    h, w = image.shape[:2]
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
        full_name = class_name(names, cls_id)
        label = short_label(full_name, label_style)
        color = class_color_bgr(full_name, cls_id)
        draw_box_with_label(
            image,
            [x1, y1, x2, y2],
            label,
            color=color,
            show_label=show_labels,
        )
    return image


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


def letter_label(index: int) -> str:
    """Return paper-style labels: (a), (b), ..."""
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if index < len(alphabet):
        return f"({alphabet[index]})"
    return f"({index + 1})"


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


def read_original_image(image_path: Path) -> np.ndarray:
    """Read the original image in BGR format."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    return image


def run_one_model(
    model: YOLO,
    image_path: Path,
    layer_index: int,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    label_style: str,
    show_labels: bool,
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
    det_bgr = draw_predictions(result, label_style=label_style, show_labels=show_labels)
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


def read_marker_json(path: Optional[Path]) -> Dict[str, object]:
    """Read optional manual miss/false-positive markers."""
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_markers(marker_data: Dict[str, object], image_stem: str, model_name: str) -> Dict[str, object]:
    """Return markers for one image/model, supporting exact and safe-name keys."""
    image_data = marker_data.get(image_stem, {})
    if not isinstance(image_data, dict):
        return {}
    model_data = image_data.get(model_name) or image_data.get(safe_name(model_name)) or {}
    return model_data if isinstance(model_data, dict) else {}


def save_detection_figure(
    out_path: Path,
    original_bgr: np.ndarray,
    detections: List[np.ndarray],
    det_model_names: List[str],
    tile_width: int,
) -> None:
    """Save a paper-style detection comparison figure."""
    panels: List[np.ndarray] = []
    original_panel = resize_to_width(original_bgr, tile_width)
    panels.append(add_caption(original_panel, f"{letter_label(0)} Original"))
    for i, (det, name) in enumerate(zip(detections, det_model_names), start=1):
        det_panel = resize_to_width(det, tile_width)
        panels.append(add_caption(det_panel, f"{letter_label(i)} {name}"))
    figure = make_grid([panels], gap=14)
    cv2.imwrite(str(out_path), figure)


def save_feature_heatmap_figure(
    out_path: Path,
    original_bgr: np.ndarray,
    features: List[np.ndarray],
    overlays: List[np.ndarray],
    det_model_names: List[str],
    tile_width: int,
) -> None:
    """Save a two-row feature-map and heatmap-overlay comparison figure."""
    original_panel = resize_to_width(original_bgr, tile_width)
    first_row: List[np.ndarray] = [add_title(original_panel, "Original")]
    second_row: List[np.ndarray] = [add_title(original_panel, "Original")]

    for feature, overlay, name in zip(features, overlays, det_model_names):
        feature_panel = resize_to_width(feature, tile_width)
        overlay_panel = resize_to_width(overlay, tile_width)
        first_row.append(add_title(feature_panel, f"{name} | Feature map"))
        second_row.append(add_title(overlay_panel, f"{name} | Heatmap overlay"))

    figure = make_grid([first_row, second_row], gap=10)
    cv2.imwrite(str(out_path), figure)


def save_asset_set(
    asset_dir: Path,
    original_bgr: np.ndarray,
    original_annotated: np.ndarray,
    detections: List[np.ndarray],
    features: List[np.ndarray],
    overlays: List[np.ndarray],
    det_model_names: List[str],
    tile_width: int,
) -> None:
    """Save individual materials for manual PPT composition."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(asset_dir / "00_original_clean.jpg"), resize_to_width(original_bgr, tile_width))
    cv2.imwrite(str(asset_dir / "00_original_annotated.jpg"), resize_to_width(original_annotated, tile_width))
    for i, (name, det, feature, overlay) in enumerate(zip(det_model_names, detections, features, overlays), start=1):
        prefix = f"{i:02d}_{safe_name(name)}"
        cv2.imwrite(str(asset_dir / f"{prefix}_detection.jpg"), resize_to_width(det, tile_width))
        cv2.imwrite(str(asset_dir / f"{prefix}_feature_map.jpg"), resize_to_width(feature, tile_width))
        cv2.imwrite(str(asset_dir / f"{prefix}_heatmap_overlay.jpg"), resize_to_width(overlay, tile_width))


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
    parser.add_argument("--mark-json", default=None, type=Path, help="Optional manual miss/false-positive marker JSON.")
    parser.add_argument("--label-style", default="short", choices=["short", "full"], help="Detection label style.")
    parser.add_argument("--hide-det-labels", action="store_true", help="Draw boxes without class/conf text labels.")
    parser.add_argument(
        "--make-comparison",
        action="store_true",
        help="Also save stitched comparison figures. By default only individual assets are saved.",
    )
    parser.add_argument(
        "--save-old-combined",
        action="store_true",
        help="Also save the old three-row combined figure named *_module_effects.jpg.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.source)[: args.max_images]
    specs = [parse_model_spec(s) for s in args.model]
    marker_data = read_marker_json(args.mark_json)

    loaded = []
    all_weights: Dict[str, object] = {}
    for spec in specs:
        model = load_yolo(spec)
        model_name = spec["name"]
        layer_index = int(spec.get("layer", "4"))
        loaded.append((model_name, model, layer_index))
        all_weights[model_name] = extract_weighted_concat_weights(model)
    write_json(args.out / "weighted_concat_weights.json", all_weights)
    gt_names = get_model_names(loaded[0][1]) if loaded else {}

    for image_path in image_paths:
        stem = image_path.stem
        per_image_dir = args.out / f"{stem}_assets"
        per_image_dir.mkdir(parents=True, exist_ok=True)

        original_bgr = read_original_image(image_path)
        original_annotated = draw_ground_truth(
            original_bgr,
            image_path,
            gt_names,
            label_style=args.label_style,
            show_labels=not args.hide_det_labels,
        )
        det_model_names: List[str] = []
        detections: List[np.ndarray] = []
        features: List[np.ndarray] = []
        overlays: List[np.ndarray] = []

        for model_name, model, layer_index in loaded:
            print(f"[{stem}] {model_name}: layer={layer_index}")
            det, feature, overlay = run_one_model(
                model,
                image_path,
                layer_index,
                args.imgsz,
                args.conf,
                args.iou,
                args.device,
                label_style=args.label_style,
                show_labels=not args.hide_det_labels,
            )
            det = draw_error_markers(det, get_markers(marker_data, stem, model_name))
            det_model_names.append(model_name)
            detections.append(det)
            features.append(feature)
            overlays.append(overlay)

            if args.save_single:
                name = safe_name(model_name)
                cv2.imwrite(str(per_image_dir / f"{name}_detection.jpg"), resize_to_width(det, args.tile_width))
                cv2.imwrite(str(per_image_dir / f"{name}_feature_map.jpg"), resize_to_width(feature, args.tile_width))
                cv2.imwrite(str(per_image_dir / f"{name}_heatmap_overlay.jpg"), resize_to_width(overlay, args.tile_width))

        save_asset_set(
            per_image_dir,
            original_bgr,
            original_annotated,
            detections,
            features,
            overlays,
            det_model_names,
            args.tile_width,
        )

        if args.make_comparison:
            save_detection_figure(
                args.out / f"{stem}_detection_comparison.jpg",
                original_annotated,
                detections,
                det_model_names,
                args.tile_width,
            )
            save_feature_heatmap_figure(
                args.out / f"{stem}_feature_heatmap_comparison.jpg",
                original_bgr,
                features,
                overlays,
                det_model_names,
                args.tile_width,
            )

        if args.save_old_combined:
            detection_row = [
                add_title(resize_to_width(det, args.tile_width), f"{name} | Detection")
                for det, name in zip(detections, det_model_names)
            ]
            feature_row = [
                add_title(resize_to_width(feature, args.tile_width), f"{name} | Feature map")
                for feature, name in zip(features, det_model_names)
            ]
            overlay_row = [
                add_title(resize_to_width(overlay, args.tile_width), f"{name} | Heatmap overlay")
                for overlay, name in zip(overlays, det_model_names)
            ]
            montage = make_grid([detection_row, feature_row, overlay_row])
            cv2.imwrite(str(args.out / f"{stem}_module_effects.jpg"), montage)

    print(f"Saved visualizations to: {args.out.resolve()}")
    print(f"Saved WeightedConcat weights to: {(args.out / 'weighted_concat_weights.json').resolve()}")


if __name__ == "__main__":
    main()
