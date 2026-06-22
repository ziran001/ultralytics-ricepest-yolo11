import os
import re
import csv
import yaml
import shutil
import argparse
from collections import defaultdict


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


SPECIMEN_RANGES = [
    (2967, 4487),
    (7647, 8521),
]


def load_yaml(path):
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data if data is not None else {}


def get_class_names(data):
    names = data.get("names", {})

    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    return {}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_leading_number(filename):
    """
    从文件名前面提取数字。
    例如:
    2967.jpg -> 2967
    2967_xxx.jpg -> 2967
    0002967_xxx.jpg -> 2967
    abc2967.jpg -> None
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^(\d+)", stem)

    if m is None:
        return None

    return int(m.group(1))


def is_specimen_file(filename):
    num = parse_leading_number(filename)

    if num is None:
        return False, None

    for start, end in SPECIMEN_RANGES:
        if start <= num <= end:
            return True, num

    return False, num


def find_dataset_splits(dataset_root):
    """
    支持 YOLO 常见结构:
    dataset/
    ├── train/images
    ├── train/labels
    ├── val/images
    ├── val/labels
    └── test/images
    """
    splits = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_root, split, "images")
        lab_dir = os.path.join(dataset_root, split, "labels")

        if os.path.exists(img_dir):
            splits.append((split, img_dir, lab_dir))

    return splits


def find_images(image_dir):
    files = []

    if not os.path.exists(image_dir):
        return files

    for name in os.listdir(image_dir):
        ext = os.path.splitext(name)[1].lower()

        if ext in IMAGE_EXTS:
            files.append(os.path.join(image_dir, name))

    return sorted(files)


def read_yolo_label(label_path):
    labels = []

    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line in lines:
        parts = line.split()

        if len(parts) < 5:
            continue

        try:
            cls_id = int(float(parts[0]))
        except Exception:
            continue

        labels.append(cls_id)

    return labels


def safe_remove_or_move(src_path, dataset_root, removed_root, action):
    """
    action:
    - move: 移动到 removed_root，保留相对路径
    - delete: 永久删除
    """
    if not os.path.exists(src_path):
        return False

    if action == "delete":
        os.remove(src_path)
        return True

    rel_path = os.path.relpath(src_path, dataset_root)
    dst_path = os.path.join(removed_root, rel_path)

    ensure_dir(os.path.dirname(dst_path))
    shutil.move(src_path, dst_path)

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/root/ultralytics-8.3.27/RicePest-30-YOLO",
        help="RicePest-30-YOLO 数据集路径"
    )

    parser.add_argument(
        "--action",
        type=str,
        default="move",
        choices=["move", "delete"],
        help="move 表示移到备份目录；delete 表示永久删除"
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正执行删除/移动；不加则只做 dry-run 统计"
    )

    parser.add_argument(
        "--removed_root",
        type=str,
        default=None,
        help="action=move 时，标本图片移动到这里；默认自动生成"
    )

    parser.add_argument(
        "--record_csv",
        type=str,
        default=None,
        help="删除记录 CSV 路径；默认保存在 dataset_root 下"
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"找不到数据集目录: {dataset_root}")

    data_yaml = os.path.join(dataset_root, "data.yaml")
    data = load_yaml(data_yaml)
    class_names = get_class_names(data)

    if args.removed_root is None:
        removed_root = dataset_root.rstrip("/") + "_removed_specimen"
    else:
        removed_root = args.removed_root

    if args.record_csv is None:
        record_csv = os.path.join(dataset_root, "removed_specimen_records.csv")
    else:
        record_csv = args.record_csv

    splits = find_dataset_splits(dataset_root)

    if len(splits) == 0:
        raise RuntimeError(
            f"没有在 {dataset_root} 下找到 train/val/test/images 结构，请检查数据集目录。"
        )

    print("\n========== 当前配置 ==========")
    print(f"dataset_root : {dataset_root}")
    print(f"data_yaml    : {data_yaml}")
    print(f"action       : {args.action}")
    print(f"execute      : {args.execute}")
    print(f"removed_root : {removed_root}")
    print(f"record_csv   : {record_csv}")
    print(f"删除编号范围 : {SPECIMEN_RANGES}")

    print("\n========== 类别列表 ==========")
    if class_names:
        for cid in sorted(class_names.keys()):
            print(f"{cid}: {class_names[cid]}")
    else:
        print("未从 data.yaml 读取到 names，后续只显示 class_id。")

    records = []

    total_images_to_remove = 0
    total_labels_to_remove = 0
    missing_label_count = 0

    # 每类出现于多少张将删除的图片
    class_image_count = defaultdict(int)

    # 每类删除多少个实例
    class_instance_count = defaultdict(int)

    # 按 split 统计
    split_image_count = defaultdict(int)
    split_label_count = defaultdict(int)

    for split, img_dir, lab_dir in splits:
        image_paths = find_images(img_dir)

        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            is_specimen, number = is_specimen_file(img_name)

            if not is_specimen:
                continue

            stem = os.path.splitext(img_name)[0]
            label_path = os.path.join(lab_dir, stem + ".txt")

            cls_ids = read_yolo_label(label_path)

            total_images_to_remove += 1
            split_image_count[split] += 1

            if os.path.exists(label_path):
                total_labels_to_remove += 1
                split_label_count[split] += 1
            else:
                missing_label_count += 1

            unique_cls_ids = sorted(set(cls_ids))

            for cid in unique_cls_ids:
                class_image_count[cid] += 1

            for cid in cls_ids:
                class_instance_count[cid] += 1

            records.append({
                "split": split,
                "number": number,
                "image_path": img_path,
                "label_path": label_path,
                "label_exists": os.path.exists(label_path),
                "class_ids": " ".join(map(str, unique_cls_ids)),
                "class_names": " | ".join(class_names.get(cid, str(cid)) for cid in unique_cls_ids),
                "instance_count": len(cls_ids)
            })

    print("\n========== Dry-run 统计 ==========")
    print(f"待删除图片数量       : {total_images_to_remove}")
    print(f"待删除 label 数量    : {total_labels_to_remove}")
    print(f"缺失 label 的图片数  : {missing_label_count}")

    print("\n按 split 统计：")
    for split in sorted(split_image_count.keys()):
        print(
            f"{split}: 图片 {split_image_count[split]} 张, "
            f"label {split_label_count[split]} 个"
        )

    print("\n========== 删除图片包含的类别统计 ==========")
    print("class_id | class_name | image_count | instance_count")

    all_class_ids = sorted(set(list(class_image_count.keys()) + list(class_instance_count.keys())))

    for cid in all_class_ids:
        cname = class_names.get(cid, str(cid))
        img_count = class_image_count.get(cid, 0)
        inst_count = class_instance_count.get(cid, 0)
        print(f"{cid} | {cname} | {img_count} | {inst_count}")

    ensure_dir(os.path.dirname(record_csv))

    with open(record_csv, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "split",
            "number",
            "image_path",
            "label_path",
            "label_exists",
            "class_ids",
            "class_names",
            "instance_count"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n删除记录 CSV 已保存: {record_csv}")

    if not args.execute:
        print("\n当前是 dry-run，没有真正删除文件。")
        print("确认无误后，加 --execute 执行。")
        return

    print("\n========== 开始执行 ==========")

    removed_images = 0
    removed_labels = 0

    if args.action == "move":
        ensure_dir(removed_root)

    for r in records:
        img_path = r["image_path"]
        label_path = r["label_path"]

        if safe_remove_or_move(
            src_path=img_path,
            dataset_root=dataset_root,
            removed_root=removed_root,
            action=args.action
        ):
            removed_images += 1

        if os.path.exists(label_path):
            if safe_remove_or_move(
                src_path=label_path,
                dataset_root=dataset_root,
                removed_root=removed_root,
                action=args.action
            ):
                removed_labels += 1

    print("\n========== 执行完成 ==========")
    print(f"已处理图片数量 : {removed_images}")
    print(f"已处理标签数量 : {removed_labels}")

    if args.action == "move":
        print(f"被移走的文件保存在: {removed_root}")
    else:
        print("已永久删除。")

    print(f"删除记录 CSV: {record_csv}")


if __name__ == "__main__":
    main()