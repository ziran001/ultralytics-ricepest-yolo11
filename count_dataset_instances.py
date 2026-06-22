import os
import csv
import yaml
import argparse
from collections import defaultdict


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="YOLO 数据集路径，例如 RicePest-30-YOLO_pruned_smallcls"
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="需要统计的 split，例如 train,val 或 train,val,test"
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="输出 CSV 路径"
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]

    data_yaml = os.path.join(dataset_root, "data.yaml")
    data = load_yaml(data_yaml)
    id_to_name = get_class_names(data)

    if not id_to_name:
        raise RuntimeError(f"无法从 data.yaml 读取类别 names: {data_yaml}")

    total_image_count = 0
    total_label_count = 0
    total_empty_label_images = 0
    total_missing_label_images = 0
    total_instance_count = 0

    class_image_count = defaultdict(int)
    class_instance_count = defaultdict(int)

    split_class_image_count = defaultdict(lambda: defaultdict(int))
    split_class_instance_count = defaultdict(lambda: defaultdict(int))
    split_summary = {}

    for split in splits:
        image_dir = os.path.join(dataset_root, split, "images")
        label_dir = os.path.join(dataset_root, split, "labels")

        image_paths = find_images(image_dir)

        split_images = 0
        split_labels = 0
        split_empty = 0
        split_missing = 0
        split_instances = 0

        for img_path in image_paths:
            split_images += 1
            total_image_count += 1

            stem = image_stem(img_path)
            label_path = os.path.join(label_dir, stem + ".txt")

            if not os.path.exists(label_path):
                split_missing += 1
                total_missing_label_images += 1
                continue

            split_labels += 1
            total_label_count += 1

            cls_ids = read_yolo_label(label_path)

            if len(cls_ids) == 0:
                split_empty += 1
                total_empty_label_images += 1

            split_instances += len(cls_ids)
            total_instance_count += len(cls_ids)

            unique_cls_ids = sorted(set(cls_ids))

            for cid in unique_cls_ids:
                class_image_count[cid] += 1
                split_class_image_count[split][cid] += 1

            for cid in cls_ids:
                class_instance_count[cid] += 1
                split_class_instance_count[split][cid] += 1

        split_summary[split] = {
            "images": split_images,
            "labels": split_labels,
            "empty_label_images": split_empty,
            "missing_label_images": split_missing,
            "instances": split_instances,
        }

    print("\n========== 数据集统计 ==========")
    print(f"dataset_root          : {dataset_root}")
    print(f"data_yaml             : {data_yaml}")
    print(f"统计 splits           : {splits}")
    print(f"总图片数              : {total_image_count}")
    print(f"总 label 文件数        : {total_label_count}")
    print(f"空标签图片数          : {total_empty_label_images}")
    print(f"缺失 label 图片数      : {total_missing_label_images}")
    print(f"总实例数              : {total_instance_count}")

    print("\n========== 按 split 统计 ==========")
    for split, s in split_summary.items():
        print(
            f"{split}: images={s['images']}, "
            f"labels={s['labels']}, "
            f"instances={s['instances']}, "
            f"empty_labels={s['empty_label_images']}, "
            f"missing_labels={s['missing_label_images']}"
        )

    print("\n========== 每类统计 ==========")
    print("class_id | class_name | images | instances")

    rows = []

    for cid in sorted(id_to_name.keys()):
        row = {
            "class_id": cid,
            "class_name": id_to_name[cid],
            "total_images": class_image_count.get(cid, 0),
            "total_instances": class_instance_count.get(cid, 0),
        }

        for split in splits:
            row[f"{split}_images"] = split_class_image_count[split].get(cid, 0)
            row[f"{split}_instances"] = split_class_instance_count[split].get(cid, 0)

        rows.append(row)

        print(
            f"{cid} | {id_to_name[cid]} | "
            f"{row['total_images']} | {row['total_instances']}"
        )

    if args.out_csv is None:
        out_csv = os.path.join(dataset_root, "dataset_instance_count.csv")
    else:
        out_csv = args.out_csv

    if rows:
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n统计结果 CSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()