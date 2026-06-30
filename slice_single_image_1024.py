"""
Slice one high-resolution pest-monitoring image into 1024x1024 tiles.

The script is intended for 5104x7016 light-trap images before YOLO inference or
feature visualization. It saves every tile plus a JSON/CSV index containing the
tile coordinates in the original image, which is useful when mapping detection
boxes back to the full-resolution image.

Example:
python slice_single_image_1024.py \
  --source /root/ultralytics-8.3.27/vis_images/full_001.jpg \
  --out /root/ultralytics-8.3.27/vis_images_sliced \
  --tile-size 1024 \
  --overlap 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(source: Path) -> List[Path]:
    """Collect image paths from a single image or a directory."""
    if source.is_file():
        if source.suffix.lower() not in IMG_EXTS:
            raise ValueError(f"Unsupported image suffix: {source}")
        return [source]
    if not source.exists():
        raise FileNotFoundError(source)
    images = [p for p in sorted(source.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise FileNotFoundError(f"No images found under {source}")
    return images


def make_positions(length: int, tile_size: int, stride: int) -> List[int]:
    """Create sliding-window start positions and force the last tile to cover the image edge."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def pad_tile(tile: np.ndarray, tile_size: int, pad_value: int) -> np.ndarray:
    """Pad edge tiles to tile_size x tile_size."""
    h, w = tile.shape[:2]
    if h == tile_size and w == tile_size:
        return tile
    padded = np.full((tile_size, tile_size, 3), pad_value, dtype=tile.dtype)
    padded[:h, :w] = tile
    return padded


def image_mean(tile: np.ndarray) -> float:
    """Return mean gray value for optional blank-tile filtering."""
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def image_std(tile: np.ndarray) -> float:
    """Return gray-value standard deviation for optional blank-tile filtering."""
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    return float(gray.std())


def should_skip_tile(tile: np.ndarray, min_std: float, max_mean: float) -> bool:
    """Skip almost blank tiles if requested."""
    if min_std <= 0 and max_mean >= 255:
        return False
    return image_std(tile) < min_std or image_mean(tile) > max_mean


def slice_image(
    image_path: Path,
    out_dir: Path,
    tile_size: int,
    overlap: float,
    pad_value: int,
    image_format: str,
    min_std: float,
    max_mean: float,
) -> List[Dict[str, object]]:
    """Slice a single image and return tile metadata."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    h, w = image.shape[:2]
    stride = int(round(tile_size * (1.0 - overlap)))
    if stride <= 0:
        raise ValueError("overlap is too large; stride must be positive.")

    xs = make_positions(w, tile_size, stride)
    ys = make_positions(h, tile_size, stride)
    image_out = out_dir / image_path.stem
    image_out.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    tile_id = 0
    suffix = image_format.lower().lstrip(".")

    for y in ys:
        for x in xs:
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            raw_tile = image[y:y2, x:x2]
            tile = pad_tile(raw_tile, tile_size, pad_value)

            if should_skip_tile(tile, min_std=min_std, max_mean=max_mean):
                continue

            tile_name = f"{image_path.stem}_x{x}_y{y}_w{x2 - x}_h{y2 - y}.{suffix}"
            tile_path = image_out / tile_name
            ok = cv2.imwrite(str(tile_path), tile)
            if not ok:
                raise OSError(f"Failed to write tile: {tile_path}")

            records.append(
                {
                    "tile_id": tile_id,
                    "source_image": str(image_path),
                    "tile_path": str(tile_path),
                    "x": x,
                    "y": y,
                    "width": x2 - x,
                    "height": y2 - y,
                    "tile_size": tile_size,
                    "padded": (x2 - x) != tile_size or (y2 - y) != tile_size,
                    "original_width": w,
                    "original_height": h,
                }
            )
            tile_id += 1

    return records


def save_index(records: List[Dict[str, object]], out_dir: Path) -> Tuple[Path, Path]:
    """Save JSON and CSV tile indexes."""
    json_path = out_dir / "slice_index.json"
    csv_path = out_dir / "slice_index.csv"

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "tile_id",
        "source_image",
        "tile_path",
        "x",
        "y",
        "width",
        "height",
        "tile_size",
        "padded",
        "original_width",
        "original_height",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice high-resolution pest images into 1024x1024 tiles.")
    parser.add_argument("--source", required=True, type=Path, help="A single image or a directory of images.")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for sliced tiles.")
    parser.add_argument("--tile-size", default=1024, type=int, help="Tile size. Default: 1024.")
    parser.add_argument("--overlap", default=0.2, type=float, help="Overlap ratio in [0, 0.9). Default: 0.2.")
    parser.add_argument("--pad-value", default=114, type=int, help="Padding value for edge tiles. Default: 114.")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], help="Output tile image format.")
    parser.add_argument("--min-std", default=0.0, type=float, help="Skip low-texture tiles below this gray std. Default off.")
    parser.add_argument("--max-mean", default=255.0, type=float, help="Skip bright blank tiles above this mean. Default off.")
    args = parser.parse_args()

    if not 0 <= args.overlap < 0.9:
        raise ValueError("--overlap must be in [0, 0.9).")
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive.")
    if not 0 <= args.pad_value <= 255:
        raise ValueError("--pad-value must be in [0, 255].")

    args.out.mkdir(parents=True, exist_ok=True)
    all_records: List[Dict[str, object]] = []

    for image_path in collect_images(args.source):
        records = slice_image(
            image_path=image_path,
            out_dir=args.out,
            tile_size=args.tile_size,
            overlap=args.overlap,
            pad_value=args.pad_value,
            image_format=args.format,
            min_std=args.min_std,
            max_mean=args.max_mean,
        )
        all_records.extend(records)
        print(f"{image_path}: saved {len(records)} tiles")

    json_path, csv_path = save_index(all_records, args.out)
    print(f"Done. Total tiles: {len(all_records)}")
    print(f"JSON index: {json_path}")
    print(f"CSV index: {csv_path}")


if __name__ == "__main__":
    main()
