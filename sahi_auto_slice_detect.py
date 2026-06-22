from pathlib import Path
import argparse
import json

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use SAHI sliced inference with Ultralytics YOLO weights."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="/root/ultralytics-8.3.27/runs/train/yolo11_mixed_dataset/weights/best.pt",
        help="Path to YOLO weight file.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset_test_6.10",
        help="Folder containing images to detect.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/sahi_predict_dataset_test_6.10",
        help="Folder to save prediction results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=1024,
        help="Slice height and width.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap ratio between adjacent slices.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--postprocess-iou",
        type=float,
        default=0.5,
        help="IoU threshold for merging duplicated boxes from overlapping slices.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="none",
        choices=["none", "id", "name", "full"],
        help="Text shown on boxes: none, id, name, or full name+score.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Bounding box line thickness.",
    )
    return parser.parse_args()


def collect_images(source_dir):
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(
        p for p in Path(source_dir).rglob("*") if p.suffix.lower() in image_suffixes
    )


def read_image_as_rgb(image_path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def color_for_category(category_id):
    palette = [
        (0, 115, 255),
        (30, 170, 50),
        (255, 120, 0),
        (200, 70, 200),
        (0, 180, 190),
        (80, 80, 230),
        (180, 140, 20),
        (40, 40, 40),
    ]
    return palette[int(category_id) % len(palette)]


def make_label(obj, label_mode):
    if label_mode == "none":
        return ""
    if label_mode == "id":
        return str(obj.category.id)
    if label_mode == "name":
        return obj.category.name
    return f"{obj.category.name} {obj.score.value:.2f}"


def draw_clean_predictions(image_path, result, output_path, label_mode, line_thickness):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image for visualization: {image_path}")

    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        x1, y1, x2, y2 = map(lambda v: int(round(v)), [x1, y1, x2, y2])
        color = color_for_category(obj.category.id)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        label = make_label(obj, label_mode)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            text_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )
            label_y1 = max(0, y1 - text_h - baseline - 4)
            label_y2 = label_y1 + text_h + baseline + 4
            label_x2 = min(image.shape[1] - 1, x1 + text_w + 6)

            cv2.rectangle(image, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(
                image,
                label,
                (x1 + 3, label_y2 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

    cv2.imwrite(str(output_path), image)


def save_predictions_json(result, image_path, output_path):
    predictions = []

    for obj in result.object_prediction_list:
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        predictions.append(
            {
                "image": str(image_path),
                "category_id": int(obj.category.id),
                "category_name": obj.category.name,
                "score": float(obj.score.value),
                "bbox_xyxy": [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],
                "bbox_xywh": [
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                ],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {source_dir}")

    images = collect_images(source_dir)
    if not images:
        raise FileNotFoundError(f"No images found in: {source_dir}")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )

    print(f"Loaded weights: {args.weights}")
    print(f"Found {len(images)} image(s) in: {source_dir}")
    print(f"Saving results to: {output_dir}")

    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] Detecting: {image_path}")
        image_rgb = read_image_as_rgb(image_path)

        result = get_sliced_prediction(
            image=image_rgb,
            detection_model=detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=args.postprocess_iou,
            verbose=0,
        )

        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        # Save a clean visualization with boxes mapped back to original-image coordinates.
        draw_clean_predictions(
            image_path=image_path,
            result=result,
            output_path=image_output_dir / "prediction_clean.jpg",
            label_mode=args.label_mode,
            line_thickness=args.line_thickness,
        )

        # Save final boxes after mapping back to original-image coordinates.
        save_predictions_json(result, image_path, image_output_dir / "result.json")

    print("Done.")


if __name__ == "__main__":
    main()
