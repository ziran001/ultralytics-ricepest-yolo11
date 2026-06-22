import os
import csv
import yaml
import shutil
import argparse
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


def get_class_names(data):
    names = data.get("names", {})

    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    return {}


def parse_int_set(s):
    if s is None or str(s).strip() == "":
        return set()

    out = set()
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.add(int(x))
    return out


def parse_splits(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def find_images(image_dir):
    files = []
    if not os.path.exists(image_dir):
        return files

    for name in os.listdir(image_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(os.path.join(image_dir, name))

    return sorted(files)


def image_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def find_image_by_stem(image_dir, stem):
    for ext in IMAGE_EXTS:
        p = os.path.join(image_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def read_yolo_label(label_path):
    labels = []
    bad_lines = []

    if not os.path.exists(label_path):
        return labels, bad_lines

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]

    for line_idx, line in enumerate(lines):
        parts = line.split()

        if len(parts) < 5:
            bad_lines.append((line_idx + 1, line, "字段数不足"))
            continue

        try:
            cls_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
        except Exception:
            bad_lines.append((line_idx + 1, line, "无法解析数字"))
            continue

        labels.append([cls_id, xc, yc, w, h])

    return labels, bad_lines


def write_yolo_label(label_path, labels):
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, w, h in labels:
            f.write(f"{int(cls_id)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def analyze_dataset(dataset_root, splits, id_to_name, remove_ids):
    class_img_count = defaultdict(int)
    class_inst_count = defaultdict(int)

    split_class_img_count = defaultdict(lambda: defaultdict(int))
    split_class_inst_count = defaultdict(lambda: defaultdict(int))

    image_records = []
    bad_records = []

    total_images = 0
    total_labels = 0
    total_instances = 0

    for split in splits:
        img_dir = os.path.join(dataset_root, split, "images")
        lab_dir = os.path.join(dataset_root, split, "labels")

        if not os.path.exists(img_dir):
            print(f"[WARN] 跳过 {split}: 不存在 {img_dir}")
            continue

        image_paths = find_images(img_dir)

        for img_path in tqdm(image_paths, desc=f"Analyze {split}"):
            total_images += 1

            stem = image_stem(img_path)
            label_path = os.path.join(lab_dir, stem + ".txt")

            labels, bad_lines = read_yolo_label(label_path)

            if os.path.exists(label_path):
                total_labels += 1

            total_instances += len(labels)

            old_ids = [int(x[0]) for x in labels]
            old_unique_ids = sorted(set(old_ids))

            kept_labels = [lab for lab in labels if int(lab[0]) not in remove_ids]
            removed_labels = [lab for lab in labels if int(lab[0]) in remove_ids]

            kept_ids = [int(x[0]) for x in kept_labels]
            removed_ids = [int(x[0]) for x in removed_labels]

            for cid in old_unique_ids:
                class_img_count[cid] += 1
                split_class_img_count[split][cid] += 1

            for cid in old_ids:
                class_inst_count[cid] += 1
                split_class_inst_count[split][cid] += 1

            image_records.append({
                "split": split,
                "image_path": img_path,
                "label_path": label_path,
                "old_instance_count": len(labels),
                "kept_instance_count_after_prune": len(kept_labels),
                "removed_instance_count": len(removed_labels),
                "old_class_ids": " ".join(map(str, old_unique_ids)),
                "kept_class_ids": " ".join(map(str, sorted(set(kept_ids)))),
                "removed_class_ids": " ".join(map(str, sorted(set(removed_ids)))),
                "will_be_empty_after_prune": int(len(kept_labels) == 0),
            })

            for line_num, raw_line, reason in bad_lines:
                bad_records.append({
                    "split": split,
                    "label_path": label_path,
                    "line_num": line_num,
                    "raw_line": raw_line,
                    "reason": reason,
                })

    all_yaml_ids = sorted(id_to_name.keys())

    class_rows = []
    for cid in all_yaml_ids:
        row = {
            "old_class_id": cid,
            "class_name": id_to_name.get(cid, str(cid)),
            "remove_flag": int(cid in remove_ids),
            "total_images": class_img_count.get(cid, 0),
            "total_instances": class_inst_count.get(cid, 0),
        }

        for split in splits:
            row[f"{split}_images"] = split_class_img_count[split].get(cid, 0)
            row[f"{split}_instances"] = split_class_inst_count[split].get(cid, 0)

        class_rows.append(row)

    return {
        "class_rows": class_rows,
        "image_records": image_records,
        "bad_records": bad_records,
        "class_inst_count": class_inst_count,
        "class_img_count": class_img_count,
        "total_images": total_images,
        "total_labels": total_labels,
        "total_instances": total_instances,
    }


def write_reports(out_dir, analysis):
    ensure_dir(out_dir)

    class_csv = os.path.join(out_dir, "class_stats_before_prune.csv")
    image_csv = os.path.join(out_dir, "image_prune_plan.csv")
    bad_csv = os.path.join(out_dir, "bad_label_records.csv")

    class_rows = analysis["class_rows"]
    image_records = analysis["image_records"]
    bad_records = analysis["bad_records"]

    if class_rows:
        with open(class_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
            writer.writeheader()
            writer.writerows(class_rows)

    if image_records:
        with open(image_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(image_records[0].keys()))
            writer.writeheader()
            writer.writerows(image_records)

    if bad_records:
        with open(bad_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(bad_records[0].keys()))
            writer.writeheader()
            writer.writerows(bad_records)

    return class_csv, image_csv, bad_csv


def build_mapping(id_to_name, remove_ids, class_inst_count, drop_zero_classes=True):
    remaining_old_ids = []

    for old_id in sorted(id_to_name.keys()):
        if old_id in remove_ids:
            continue

        if drop_zero_classes and class_inst_count.get(old_id, 0) == 0:
            continue

        remaining_old_ids.append(old_id)

    old_to_new = {old_id: new_id for new_id, old_id in enumerate(remaining_old_ids)}
    new_names = [id_to_name[old_id] for old_id in remaining_old_ids]

    return old_to_new, new_names


def make_pruned_dataset(
    dataset_root,
    out_root,
    splits,
    old_to_new,
    new_names,
    remove_ids,
    drop_empty_images=False,
    overwrite=False
):
    if os.path.exists(out_root):
        if overwrite:
            print(f"删除已有输出目录: {out_root}")
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(f"输出目录已存在: {out_root}，如需覆盖请加 --overwrite")

    ensure_dir(out_root)

    modified_records = []

    for split in splits:
        src_img_dir = os.path.join(dataset_root, split, "images")
        src_lab_dir = os.path.join(dataset_root, split, "labels")

        out_img_dir = os.path.join(out_root, split, "images")
        out_lab_dir = os.path.join(out_root, split, "labels")

        ensure_dir(out_img_dir)
        ensure_dir(out_lab_dir)

        if not os.path.exists(src_img_dir):
            continue

        image_paths = find_images(src_img_dir)

        for img_path in tqdm(image_paths, desc=f"Create pruned {split}"):
            stem = image_stem(img_path)
            label_path = os.path.join(src_lab_dir, stem + ".txt")

            labels, _ = read_yolo_label(label_path)

            new_labels = []
            removed_count = 0

            for lab in labels:
                old_id = int(lab[0])

                if old_id in remove_ids:
                    removed_count += 1
                    continue

                if old_id not in old_to_new:
                    removed_count += 1
                    continue

                new_id = old_to_new[old_id]
                new_labels.append([new_id] + lab[1:5])

            if drop_empty_images and len(new_labels) == 0:
                modified_records.append({
                    "split": split,
                    "image": img_path,
                    "status": "dropped_empty_after_prune",
                    "old_instances": len(labels),
                    "new_instances": 0,
                    "removed_instances": removed_count,
                })
                continue

            dst_img = os.path.join(out_img_dir, os.path.basename(img_path))
            dst_lab = os.path.join(out_lab_dir, stem + ".txt")

            shutil.copy2(img_path, dst_img)
            write_yolo_label(dst_lab, new_labels)

            modified_records.append({
                "split": split,
                "image": img_path,
                "status": "kept",
                "old_instances": len(labels),
                "new_instances": len(new_labels),
                "removed_instances": removed_count,
            })

    data_yaml = {
        "path": out_root,
        "train": "train/images",
        "val": "val/images",
        "nc": len(new_names),
        "names": new_names,
    }

    if "test" in splits and os.path.exists(os.path.join(out_root, "test", "images")):
        data_yaml["test"] = "test/images"

    save_yaml(data_yaml, os.path.join(out_root, "data.yaml"))

    mapping_csv = os.path.join(out_root, "class_id_mapping_old_to_new.csv")
    with open(mapping_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["old_class_id", "new_class_id", "class_name"]
        )
        writer.writeheader()

        for old_id, new_id in old_to_new.items():
            writer.writerow({
                "old_class_id": old_id,
                "new_class_id": new_id,
                "class_name": new_names[new_id],
            })

    modified_csv = os.path.join(out_root, "modified_image_records.csv")
    if modified_records:
        with open(modified_csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(modified_records[0].keys()))
            writer.writeheader()
            writer.writerows(modified_records)

    return mapping_csv, modified_csv


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/root/ultralytics-8.3.27/mixed_dataset_patch1024_sam_replace300",
        help="原始数据集路径"
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default="/root/ultralytics-8.3.27/mixed_dataset_patch1024_sam_replace300_pruned",
        help="输出的新数据集路径"
    )

    parser.add_argument(
        "--remove_class_ids",
        type=str,
        default="7,10,15,16,20,22,28,29",
        help="要删除的小类 old class_id，用逗号分隔"
    )

    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="处理哪些 split，例如 train,val"
    )

    parser.add_argument(
        "--report_dir",
        type=str,
        default=None,
        help="统计报告输出目录，默认 dataset_root/prune_small_class_report"
    )

    parser.add_argument(
        "--drop_zero_classes",
        action="store_true",
        help="同时删除实例数为 0 的类别"
    )

    parser.add_argument(
        "--drop_empty_images",
        action="store_true",
        help="删除标签清空后的图片；不加则保留为空标签负样本"
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="真正生成新数据集；不加则只统计 dry-run"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true"
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root
    data_yaml = os.path.join(dataset_root, "data.yaml")

    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"找不到数据集目录: {dataset_root}")

    data = load_yaml(data_yaml)
    id_to_name = get_class_names(data)

    if not id_to_name:
        raise RuntimeError(f"无法从 data.yaml 读取 names: {data_yaml}")

    remove_ids = parse_int_set(args.remove_class_ids)
    splits = parse_splits(args.splits)

    if args.report_dir is None:
        report_dir = os.path.join(dataset_root, "prune_small_class_report")
    else:
        report_dir = args.report_dir

    print("\n========== 当前配置 ==========")
    print(f"dataset_root     : {dataset_root}")
    print(f"data_yaml        : {data_yaml}")
    print(f"out_root         : {args.out_root}")
    print(f"splits           : {splits}")
    print(f"remove_class_ids : {sorted(remove_ids)}")
    print(f"drop_zero_classes: {args.drop_zero_classes}")
    print(f"drop_empty_images: {args.drop_empty_images}")
    print(f"execute          : {args.execute}")
    print(f"report_dir       : {report_dir}")

    print("\n========== 准备删除的小类 ==========")
    for cid in sorted(remove_ids):
        print(f"{cid}: {id_to_name.get(cid, 'UNKNOWN')}")

    print("\n========== 统计 train / val 类别分布 ==========")
    analysis = analyze_dataset(
        dataset_root=dataset_root,
        splits=splits,
        id_to_name=id_to_name,
        remove_ids=remove_ids
    )

    class_csv, image_csv, bad_csv = write_reports(report_dir, analysis)

    print("\n========== 数据集总体统计 ==========")
    print(f"总图片数      : {analysis['total_images']}")
    print(f"label 文件数  : {analysis['total_labels']}")
    print(f"总实例数      : {analysis['total_instances']}")

    print("\n========== 类别统计 ==========")
    print("old_id | class_name | total_images | total_instances | remove")
    for row in analysis["class_rows"]:
        print(
            f"{row['old_class_id']} | {row['class_name']} | "
            f"{row['total_images']} | {row['total_instances']} | "
            f"{row['remove_flag']}"
        )

    old_to_new, new_names = build_mapping(
        id_to_name=id_to_name,
        remove_ids=remove_ids,
        class_inst_count=analysis["class_inst_count"],
        drop_zero_classes=args.drop_zero_classes
    )

    print("\n========== 重映射后的类别 ==========")
    for old_id, new_id in old_to_new.items():
        print(f"{old_id} -> {new_id}: {id_to_name[old_id]}")

    print(f"\n原类别数: {len(id_to_name)}")
    print(f"新类别数: {len(new_names)}")

    print("\n========== 报告文件 ==========")
    print(f"类别统计 CSV : {class_csv}")
    print(f"图片处理计划 : {image_csv}")
    print(f"异常 label   : {bad_csv}")

    if not args.execute:
        print("\n当前是 dry-run，没有生成新数据集。")
        print("确认无误后，加 --execute 生成删除小类并重映射后的数据集。")
        return

    print("\n========== 开始生成新数据集 ==========")
    mapping_csv, modified_csv = make_pruned_dataset(
        dataset_root=dataset_root,
        out_root=args.out_root,
        splits=splits,
        old_to_new=old_to_new,
        new_names=new_names,
        remove_ids=remove_ids,
        drop_empty_images=args.drop_empty_images,
        overwrite=args.overwrite
    )

    print("\n========== 完成 ==========")
    print(f"新数据集目录 : {args.out_root}")
    print(f"新 data.yaml : {os.path.join(args.out_root, 'data.yaml')}")
    print(f"类别映射 CSV : {mapping_csv}")
    print(f"图片处理记录 : {modified_csv}")


if __name__ == "__main__":
    main()