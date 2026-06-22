import os
import cv2
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import SAM


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def load_class_names(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", None)
    if names is None:
        raise ValueError("data.yaml 中没有 names 字段")

    if isinstance(names, list):
        id_to_name = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        id_to_name = {int(k): v for k, v in names.items()}
    else:
        raise ValueError("data.yaml 中 names 格式错误，应为 list 或 dict")

    name_to_id = {v: k for k, v in id_to_name.items()}
    return id_to_name, name_to_id


def find_image_path(image_dir, stem):
    for ext in IMAGE_EXTS:
        p = os.path.join(image_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h, expand=0.08):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h

    bw = x2 - x1
    bh = y2 - y1

    x1 -= bw * expand
    y1 -= bh * expand
    x2 += bw * expand
    y2 += bh * expand

    x1 = int(max(0, round(x1)))
    y1 = int(max(0, round(y1)))
    x2 = int(min(img_w - 1, round(x2)))
    y2 = int(min(img_h - 1, round(y2)))

    return [x1, y1, x2, y2]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/root/datasets_2",
        help="你的个人数据集根目录"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="建议只用 train 提取 mask，避免 val 泄漏"
    )

    parser.add_argument(
        "--target_class_name",
        type=str,
        default="Rice plant hopper",
        help="目标类别名称"
    )

    parser.add_argument(
        "--target_class_id",
        type=int,
        default=None,
        help="如果类别名匹配不上，可以手动指定类别 ID"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/root/hopper_sam_masks_datasets2",
        help="mask 输出目录"
    )

    parser.add_argument(
        "--sam_model",
        type=str,
        default="sam_b.pt",
        help="SAM 模型，可选 sam_b.pt / sam_l.pt"
    )

    parser.add_argument(
        "--box_expand",
        type=float,
        default=0.08,
        help="YOLO 框扩张比例，小目标建议 0.05~0.15"
    )

    parser.add_argument(
        "--min_mask_area",
        type=int,
        default=8,
        help="过滤过小 mask"
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root
    data_yaml = os.path.join(dataset_root, "data.yaml")
    image_dir = os.path.join(dataset_root, args.split, "images")
    label_dir = os.path.join(dataset_root, args.split, "labels")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"找不到 data.yaml: {data_yaml}")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"找不到 images 目录: {image_dir}")

    if not os.path.exists(label_dir):
        raise FileNotFoundError(f"找不到 labels 目录: {label_dir}")

    id_to_name, name_to_id = load_class_names(data_yaml)

    print("\n========== data.yaml 类别列表 ==========")
    for cid, cname in id_to_name.items():
        print(f"{cid}: {cname}")

    if args.target_class_id is not None:
        target_class_id = args.target_class_id
        target_class_name = id_to_name.get(target_class_id, str(target_class_id))
    else:
        if args.target_class_name not in name_to_id:
            raise ValueError(
                f"\n没有找到类别名: {args.target_class_name}\n"
                f"请检查 data.yaml 里的类别名，或者用 --target_class_id 手动指定。"
            )
        target_class_id = name_to_id[args.target_class_name]
        target_class_name = args.target_class_name

    print("\n========== 当前配置 ==========")
    print(f"dataset_root    : {dataset_root}")
    print(f"split           : {args.split}")
    print(f"image_dir       : {image_dir}")
    print(f"label_dir       : {label_dir}")
    print(f"target_class_id : {target_class_id}")
    print(f"target_class    : {target_class_name}")
    print(f"out_dir         : {args.out_dir}")
    print(f"sam_model       : {args.sam_model}")
    print(f"box_expand      : {args.box_expand}")

    mask_dir = os.path.join(args.out_dir, "masks")
    rgba_dir = os.path.join(args.out_dir, "rgba")
    preview_dir = os.path.join(args.out_dir, "preview")

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(rgba_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    print("\n正在加载 SAM 模型...")
    sam = SAM(args.sam_model)

    label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.lower().endswith(".txt")
    ])

    records = []
    total_label_files = 0
    total_target_boxes = 0
    success_masks = 0

    for label_file in tqdm(label_files, desc="Reading YOLO labels and extracting masks"):
        total_label_files += 1

        stem = os.path.splitext(label_file)[0]
        label_path = os.path.join(label_dir, label_file)
        image_path = find_image_path(image_dir, stem)

        if image_path is None:
            print(f"警告：找不到对应图片: {stem}")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：图片读取失败: {image_path}")
            continue

        img_h, img_w = img.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        obj_index = 0

        for line_index, line in enumerate(lines):
            parts = line.split()

            if len(parts) < 5:
                continue

            cls_id = int(float(parts[0]))

            if cls_id != target_class_id:
                continue

            total_target_boxes += 1

            xc, yc, bw, bh = map(float, parts[1:5])

            box = yolo_to_xyxy(
                xc, yc, bw, bh,
                img_w, img_h,
                expand=args.box_expand
            )

            x1, y1, x2, y2 = box

            if x2 <= x1 or y2 <= y1:
                continue

            try:
                # 注意：这里用 YOLO 框作为 SAM 的 box prompt
                results = sam(image_path, bboxes=[box], verbose=False)
            except Exception as e:
                print(f"\nSAM 处理失败: {image_path}, box={box}, error={e}")
                continue

            if not results or results[0].masks is None:
                continue

            mask_data = results[0].masks.data

            if mask_data is None or len(mask_data) == 0:
                continue

            mask = mask_data[0].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255

            # 限制 mask 只保留在扩张后的 YOLO 框内，防止 SAM 分到其他目标
            bbox_region = np.zeros_like(mask, dtype=np.uint8)
            bbox_region[y1:y2 + 1, x1:x2 + 1] = 255
            mask = cv2.bitwise_and(mask, bbox_region)

            ys, xs = np.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            mask_area = int((mask > 0).sum())

            if mask_area < args.min_mask_area:
                continue

            mx1, my1 = int(xs.min()), int(ys.min())
            mx2, my2 = int(xs.max()), int(ys.max())

            crop_bgr = img[my1:my2 + 1, mx1:mx2 + 1]
            crop_mask = mask[my1:my2 + 1, mx1:mx2 + 1]

            if crop_bgr.size == 0 or crop_mask.size == 0:
                continue

            crop_bgra = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2BGRA)
            crop_bgra[:, :, 3] = crop_mask

            base_name = f"{stem}_hopper_{obj_index:03d}"

            mask_out = os.path.join(mask_dir, base_name + "_mask.png")
            rgba_out = os.path.join(rgba_dir, base_name + "_rgba.png")
            preview_out = os.path.join(preview_dir, base_name + "_preview.jpg")

            cv2.imwrite(mask_out, crop_mask)
            cv2.imwrite(rgba_out, crop_bgra)

            preview = img.copy()
            green_mask = np.zeros_like(img, dtype=np.uint8)
            green_mask[:, :, 1] = mask

            preview = cv2.addWeighted(preview, 0.75, green_mask, 0.25, 0)
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                preview,
                f"{target_class_name}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

            cv2.imwrite(preview_out, preview)

            records.append({
                "image_path": image_path,
                "label_path": label_path,
                "class_id": cls_id,
                "class_name": target_class_name,
                "yolo_xc": xc,
                "yolo_yc": yc,
                "yolo_w": bw,
                "yolo_h": bh,
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "mask_x1": mx1,
                "mask_y1": my1,
                "mask_x2": mx2,
                "mask_y2": my2,
                "mask_area": mask_area,
                "mask_path": mask_out,
                "rgba_path": rgba_out,
                "preview_path": preview_out
            })

            obj_index += 1
            success_masks += 1

    csv_path = os.path.join(args.out_dir, "hopper_mask_records.csv")
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n========== 完成 ==========")
    print(f"读取 label 文件数      : {total_label_files}")
    print(f"目标类别标注框数量    : {total_target_boxes}")
    print(f"成功提取 mask 数量    : {success_masks}")
    print(f"输出目录              : {args.out_dir}")
    print(f"透明虫体目录 rgba     : {rgba_dir}")
    print(f"mask 目录             : {mask_dir}")
    print(f"预览图目录 preview    : {preview_dir}")
    print(f"记录 CSV              : {csv_path}")


if __name__ == "__main__":
    main()