import os
import re
import csv
import argparse
from collections import defaultdict


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

SPECIMEN_RANGES = [
    (2967, 4487),
    (7647, 8521),
]


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


def in_ranges(num, ranges):
    for start, end in ranges:
        if start <= num <= end:
            return True
    return False


def expected_numbers(ranges):
    nums = []
    for start, end in ranges:
        nums.extend(list(range(start, end + 1)))
    return nums


def find_images(dataset_root):
    results = []

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(dataset_root, split, "images")

        if not os.path.exists(img_dir):
            continue

        for name in os.listdir(img_dir):
            ext = os.path.splitext(name)[1].lower()

            if ext not in IMAGE_EXTS:
                continue

            num = parse_leading_number(name)

            if num is None:
                continue

            if in_ranges(num, SPECIMEN_RANGES):
                results.append({
                    "split": split,
                    "number": num,
                    "filename": name,
                    "path": os.path.join(img_dir, name)
                })

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/root/ultralytics-8.3.27/RicePest-30-YOLO",
        help="RicePest-30-YOLO 数据集路径"
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="输出检查结果 CSV"
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root

    if args.out_csv is None:
        out_csv = os.path.join(dataset_root, "specimen_missing_number_check.csv")
    else:
        out_csv = args.out_csv

    records = find_images(dataset_root)

    expected = expected_numbers(SPECIMEN_RANGES)
    expected_set = set(expected)

    number_to_records = defaultdict(list)

    for r in records:
        number_to_records[r["number"]].append(r)

    present_numbers = sorted(number_to_records.keys())
    missing_numbers = sorted(expected_set - set(present_numbers))

    duplicate_numbers = {
        num: recs for num, recs in number_to_records.items()
        if len(recs) > 1
    }

    split_count = defaultdict(int)
    range_count = defaultdict(int)

    for r in records:
        split_count[r["split"]] += 1

        for start, end in SPECIMEN_RANGES:
            if start <= r["number"] <= end:
                range_count[f"{start}-{end}"] += 1

    print("\n========== 理论数量 ==========")
    for start, end in SPECIMEN_RANGES:
        print(f"{start}-{end}: {end - start + 1} 个编号")

    print(f"理论总编号数: {len(expected)}")

    print("\n========== 实际找到 ==========")
    print(f"实际图片数量: {len(records)}")
    print(f"实际唯一编号数量: {len(present_numbers)}")

    print("\n按范围统计：")
    for k in sorted(range_count.keys()):
        print(f"{k}: {range_count[k]} 张图片")

    print("\n按 split 统计：")
    for split in sorted(split_count.keys()):
        print(f"{split}: {split_count[split]} 张图片")

    print("\n========== 缺失编号 ==========")
    print(f"缺失编号数量: {len(missing_numbers)}")

    if missing_numbers:
        print("缺失编号如下：")
        print(", ".join(map(str, missing_numbers)))
    else:
        print("没有缺失编号。")

    print("\n========== 重复编号 ==========")
    print(f"重复编号数量: {len(duplicate_numbers)}")

    if duplicate_numbers:
        for num, recs in sorted(duplicate_numbers.items()):
            print(f"\n编号 {num} 出现 {len(recs)} 次:")
            for r in recs:
                print(f"  [{r['split']}] {r['filename']}")
    else:
        print("没有重复编号。")

    # 输出 CSV
    rows = []

    for num in expected:
        recs = number_to_records.get(num, [])

        if len(recs) == 0:
            rows.append({
                "number": num,
                "status": "missing",
                "count": 0,
                "split": "",
                "filename": "",
                "path": ""
            })
        else:
            status = "duplicate" if len(recs) > 1 else "present"

            for r in recs:
                rows.append({
                    "number": num,
                    "status": status,
                    "count": len(recs),
                    "split": r["split"],
                    "filename": r["filename"],
                    "path": r["path"]
                })

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["number", "status", "count", "split", "filename", "path"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n检查结果 CSV 已保存: {out_csv}")


if __name__ == "__main__":
    main()