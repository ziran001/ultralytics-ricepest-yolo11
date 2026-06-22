import os
import cv2
import yaml
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def find_images(image_dir):
    if not os.path.exists(image_dir):
        return []

    files = []
    for name in os.listdir(image_dir):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
            files.append(os.path.join(image_dir, name))

    return sorted(files)


def image_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def read_yolo_labels(label_path):
    labels = []

    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()

        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])

        labels.append([cls_id, xc, yc, w, h])

    return labels


def write_yolo_labels(label_path, labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, w, h in labels:
            xc = float(np.clip(xc, 0, 1))
            yc = float(np.clip(yc, 0, 1))
            w = float(np.clip(w, 1e-6, 1))
            h = float(np.clip(h, 1e-6, 1))

            f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def yolo_to_xyxy(label, img_w, img_h):
    cls_id, xc, yc, w, h = label

    x1 = (xc - w / 2.0) * img_w
    y1 = (yc - h / 2.0) * img_h
    x2 = (xc + w / 2.0) * img_w
    y2 = (yc + h / 2.0) * img_h

    return [cls_id, x1, y1, x2, y2]


def xyxy_to_yolo(cls_id, x1, y1, x2, y2, patch_w, patch_h):
    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        return None

    xc = (x1 + x2) / 2.0 / patch_w
    yc = (y1 + y2) / 2.0 / patch_h
    w = bw / patch_w
    h = bh / patch_h

    if w <= 0 or h <= 0:
        return None

    return [cls_id, xc, yc, w, h]


def generate_windows(img_w, img_h, tile_size=1024, overlap=0.2):
    """
    生成滑窗坐标。
    保证覆盖到图像边缘。
    """
    step = int(tile_size * (1.0 - overlap))
    step = max(1, step)

    if img_w <= tile_size:
        xs = [0]
    else:
        xs = list(range(0, img_w - tile_size + 1, step))
        last_x = img_w - tile_size
        if xs[-1] != last_x:
            xs.append(last_x)

    if img_h <= tile_size:
        ys = [0]
    else:
        ys = list(range(0, img_h - tile_size + 1, step))
        last_y = img_h - tile_size
        if ys[-1] != last_y:
            ys.append(last_y)

    windows = []

    for y in ys:
        for x in xs:
            x1 = x
            y1 = y
            x2 = min(x + tile_size, img_w)
            y2 = min(y + tile_size, img_h)
            windows.append([x1, y1, x2, y2])

    return windows


def clip_boxes_to_window(
    abs_boxes,
    window,
    min_visibility=0.25,
    min_box_size=3,
    allowed_class_ids=None
):
    """
    将原图中的绝对坐标框裁剪到 patch 内，并转换成 patch 内 YOLO 标签。
    类别 ID 不变，因为你已经提前重映射好了。
    """
    wx1, wy1, wx2, wy2 = window
    patch_w = wx2 - wx1
    patch_h = wy2 - wy1

    patch_labels = []

    for box in abs_boxes:
        cls_id, x1, y1, x2, y2 = box

        if allowed_class_ids is not None and cls_id not in allowed_class_ids:
            continue

        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        box_area = box_w * box_h

        if box_area <= 0:
            continue

        ix1 = max(x1, wx1)
        iy1 = max(y1, wy1)
        ix2 = min(x2, wx2)
        iy2 = min(y2, wy2)

        inter_w = max(0, ix2 - ix1)
        inter_h = max(0, iy2 - iy1)
        inter_area = inter_w * inter_h

        if inter_area <= 0:
            continue

        visibility = inter_area / box_area

        if visibility < min_visibility:
            continue

        px1 = ix1 - wx1
        py1 = iy1 - wy1
        px2 = ix2 - wx1
        py2 = iy2 - wy1

        if px2 - px1 < min_box_size or py2 - py1 < min_box_size:
            continue

        yolo_label = xyxy_to_yolo(
            cls_id,
            px1,
            py1,
            px2,
            py2,
            patch_w,
            patch_h
        )

        if yolo_label is not None:
            patch_labels.append(yolo_label)

    return patch_labels


def draw_preview(patch, labels, out_path):
    """
    生成切片预览图。
    红框：class_id=30，稻飞虱
    绿框：其他类别
    """
    vis = patch.copy()
    h, w = vis.shape[:2]

    for cls_id, xc, yc, bw, bh in labels:
        x1 = int((xc - bw / 2.0) * w)
        y1 = int((yc - bh / 2.0) * h)
        x2 = int((xc + bw / 2.0) * w)
        y2 = int((yc + bh / 2.0) * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        color = (0, 0, 255) if int(cls_id) == 30 else (0, 255, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            str(int(cls_id)),
            (x1, max(0, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            lineType=cv2.LINE_AA
        )

    cv2.imwrite(out_path, vis)


def parse_allowed_class_ids(s):
    if s is None or s.strip() == "":
        return None

    out = set()
    for x in s.split(","):
        x = x.strip()
        if x:
            out.add(int(x))

    return out


def make_data_yaml(src_root, out_root, names_yaml=None):
    """
    生成 data.yaml。
    如果提供 names_yaml，就优先使用 mixed_dataset 的类别体系。
    这样可以避免 labels 中有 class_id=30，但 names 只有 3 类的问题。
    """
    if names_yaml is not None and os.path.exists(names_yaml):
        data = load_yaml(names_yaml)
    else:
        src_yaml = os.path.join(src_root, "data.yaml")
        data = load_yaml(src_yaml)

    data["path"] = out_root
    data["train"] = "train/images"
    data["val"] = "val/images"

    if "test" in data:
        data.pop("test", None)

    out_yaml = os.path.join(out_root, "data.yaml")
    save_yaml(data, out_yaml)


def slice_split(
    src_root,
    out_root,
    split,
    tile_size,
    overlap,
    min_visibility,
    min_box_size,
    max_pos_per_image,
    max_empty_per_image,
    allowed_class_ids,
    jpg_quality,
    preview_num
):
    src_img_dir = os.path.join(src_root, split, "images")
    src_lab_dir = os.path.join(src_root, split, "labels")

    out_img_dir = os.path.join(out_root, split, "images")
    out_lab_dir = os.path.join(out_root, split, "labels")
    preview_dir = os.path.join(out_root, f"preview_patch_{split}")

    ensure_dir(out_img_dir)
    ensure_dir(out_lab_dir)
    ensure_dir(preview_dir)

    image_paths = find_images(src_img_dir)

    total_saved = 0
    total_labels = 0
    total_empty = 0
    class_counter = defaultdict(int)
    preview_saved = 0

    for img_path in tqdm(image_paths, desc=f"Slice {split}"):
        img = cv2.imread(img_path)

        if img is None:
            print(f"图片读取失败: {img_path}")
            continue

        img_h, img_w = img.shape[:2]
        stem = image_stem(img_path)

        label_path = os.path.join(src_lab_dir, stem + ".txt")
        yolo_labels = read_yolo_labels(label_path)
        abs_boxes = [yolo_to_xyxy(lab, img_w, img_h) for lab in yolo_labels]

        windows = generate_windows(
            img_w,
            img_h,
            tile_size=tile_size,
            overlap=overlap
        )

        pos_items = []
        empty_items = []

        for window in windows:
            patch_labels = clip_boxes_to_window(
                abs_boxes=abs_boxes,
                window=window,
                min_visibility=min_visibility,
                min_box_size=min_box_size,
                allowed_class_ids=allowed_class_ids
            )

            if len(patch_labels) > 0:
                pos_items.append((window, patch_labels))
            else:
                empty_items.append((window, patch_labels))

        if split == "train":
            if max_pos_per_image > 0 and len(pos_items) > max_pos_per_image:
                pos_items = random.sample(pos_items, max_pos_per_image)

            if max_empty_per_image > 0 and len(empty_items) > max_empty_per_image:
                empty_items = random.sample(empty_items, max_empty_per_image)
            else:
                empty_items = empty_items[:max_empty_per_image]
        else:
            # val 建议只保留有目标 patch，避免大量空背景稀释验证
            if max_pos_per_image > 0 and len(pos_items) > max_pos_per_image:
                pos_items = random.sample(pos_items, max_pos_per_image)

            empty_items = []

        selected_items = pos_items + empty_items

        for idx, (window, patch_labels) in enumerate(selected_items):
            x1, y1, x2, y2 = window
            patch = img[y1:y2, x1:x2].copy()

            if patch.size == 0:
                continue

            out_name = (
                f"patch_{split}_{stem}"
                f"_x{x1}_y{y1}_s{tile_size}_{idx:03d}.jpg"
            )

            out_img_path = os.path.join(out_img_dir, out_name)
            out_lab_path = os.path.join(
                out_lab_dir,
                os.path.splitext(out_name)[0] + ".txt"
            )

            cv2.imwrite(
                out_img_path,
                patch,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
            )

            write_yolo_labels(out_lab_path, patch_labels)

            total_saved += 1
            total_labels += len(patch_labels)

            if len(patch_labels) == 0:
                total_empty += 1

            for lab in patch_labels:
                class_counter[int(lab[0])] += 1

            if preview_saved < preview_num and len(patch_labels) > 0:
                preview_path = os.path.join(
                    preview_dir,
                    os.path.splitext(out_name)[0] + "_preview.jpg"
                )
                draw_preview(patch, patch_labels, preview_path)
                preview_saved += 1

    return total_saved, total_labels, total_empty, class_counter


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_root",
        type=str,
        default="/root/ultralytics-8.3.27/datasets_remapped",
        help="已经完成类别重映射的个人数据集"
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default="/root/ultralytics-8.3.27/datasets_remapped_patch1024",
        help="输出切片后的个人数据集"
    )

    parser.add_argument(
        "--names_yaml",
        type=str,
        default="/root/ultralytics-8.3.27/mixed_dataset 1+1/data.yaml",
        help="使用 mixed_dataset 的 data.yaml 类别体系，避免 class_id=30 越界"
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=1024,
        help="切片大小，推荐 1024 或 1280"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="滑窗重叠比例，推荐 0.2"
    )

    parser.add_argument(
        "--min_visibility",
        type=float,
        default=0.25,
        help="目标至少有多少比例落入 patch 才保留"
    )

    parser.add_argument(
        "--min_box_size",
        type=int,
        default=3,
        help="切片后目标框最小宽高，小于该值丢弃"
    )

    parser.add_argument(
        "--max_pos_per_image",
        type=int,
        default=80,
        help="每张大图最多保留多少个含目标 patch；0 表示全部保留"
    )

    parser.add_argument(
        "--max_empty_per_image",
        type=int,
        default=2,
        help="每张 train 大图最多保留多少个空背景 patch"
    )

    parser.add_argument(
        "--allowed_class_ids",
        type=str,
        default="1,2,30",
        help="允许保留的类别 ID"
    )

    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=95
    )

    parser.add_argument(
        "--preview_num",
        type=int,
        default=80
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if os.path.exists(args.out_root):
        if args.overwrite:
            print(f"删除已有输出目录: {args.out_root}")
            shutil.rmtree(args.out_root)
        else:
            raise FileExistsError(
                f"输出目录已存在: {args.out_root}\n"
                f"如需覆盖，请加 --overwrite"
            )

    ensure_dir(args.out_root)

    allowed_class_ids = parse_allowed_class_ids(args.allowed_class_ids)

    print("\n========== 当前配置 ==========")
    print(f"src_root          : {args.src_root}")
    print(f"out_root          : {args.out_root}")
    print(f"names_yaml        : {args.names_yaml}")
    print(f"tile_size         : {args.tile_size}")
    print(f"overlap           : {args.overlap}")
    print(f"min_visibility    : {args.min_visibility}")
    print(f"min_box_size      : {args.min_box_size}")
    print(f"max_pos_per_image : {args.max_pos_per_image}")
    print(f"max_empty_per_img : {args.max_empty_per_image}")
    print(f"allowed_class_ids : {allowed_class_ids}")

    print("\n========== 切片 train ==========")
    train_saved, train_labels, train_empty, train_counter = slice_split(
        src_root=args.src_root,
        out_root=args.out_root,
        split="train",
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        min_box_size=args.min_box_size,
        max_pos_per_image=args.max_pos_per_image,
        max_empty_per_image=args.max_empty_per_image,
        allowed_class_ids=allowed_class_ids,
        jpg_quality=args.jpg_quality,
        preview_num=args.preview_num
    )

    print("\n========== 切片 val ==========")
    val_saved, val_labels, val_empty, val_counter = slice_split(
        src_root=args.src_root,
        out_root=args.out_root,
        split="val",
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        min_box_size=args.min_box_size,
        max_pos_per_image=args.max_pos_per_image,
        max_empty_per_image=0,
        allowed_class_ids=allowed_class_ids,
        jpg_quality=args.jpg_quality,
        preview_num=args.preview_num
    )

    make_data_yaml(
        src_root=args.src_root,
        out_root=args.out_root,
        names_yaml=args.names_yaml
    )

    print("\n========== 完成 ==========")
    print(f"train patch 数量   : {train_saved}")
    print(f"train 空背景 patch : {train_empty}")
    print(f"train 标签数量     : {train_labels}")
    print(f"train 类别统计     : {dict(train_counter)}")
    print(f"val patch 数量     : {val_saved}")
    print(f"val 空背景 patch   : {val_empty}")
    print(f"val 标签数量       : {val_labels}")
    print(f"val 类别统计       : {dict(val_counter)}")
    print(f"输出目录           : {args.out_root}")
    print(f"data.yaml          : {os.path.join(args.out_root, 'data.yaml')}")
    print(f"train 预览图       : {os.path.join(args.out_root, 'preview_patch_train')}")
    print(f"val 预览图         : {os.path.join(args.out_root, 'preview_patch_val')}")


if __name__ == "__main__":
    main()