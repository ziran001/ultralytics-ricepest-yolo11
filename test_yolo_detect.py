#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用指定 YOLO 权重检测 dataset_test_6.10 文件夹中的图片。

运行示例：
    python test_yolo_detect.py

可选参数：
    python test_yolo_detect.py --source dataset_test_6.10 --conf 0.25 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO image detection test script")
    parser.add_argument(
        "--weights",
        type=str,
        default="/root/ultralytics-8.3.27/runs/train/yolo11_mixed_dataset/weights/best.pt",
        help="模型权重路径"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="dataset_test_6.10",
        help="待检测图片文件夹路径"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推理图片尺寸"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备，例如 0 表示第一张 GPU，cpu 表示使用 CPU"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="结果保存的父目录"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dataset_test_6.10_pred",
        help="结果保存的子目录名称"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    source_path = Path(args.source)

    if not weights_path.exists():
        raise FileNotFoundError(f"找不到权重文件：{weights_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"找不到待检测文件夹：{source_path}")

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = [
        p for p in source_path.rglob("*")
        if p.suffix.lower() in image_suffixes
    ]

    if len(image_files) == 0:
        raise RuntimeError(f"文件夹中没有找到图片：{source_path}")

    print(f"加载模型：{weights_path}")
    model = YOLO(str(weights_path))

    print(f"开始检测，共找到 {len(image_files)} 张图片")
    results = model.predict(
        source=str(source_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    save_dir = Path(args.project) / args.name
    print(f"检测完成，结果已保存到：{save_dir.resolve()}")
    print(f"共处理结果数量：{len(results)}")


if __name__ == "__main__":
    main()
