import os
import cv2
import csv
import yaml
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm


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


def image_stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def find_images(image_dir, prefix=None):
    files = []

    if not os.path.exists(image_dir):
        return files

    for name in os.listdir(image_dir):
        ext = os.path.splitext(name)[1].lower()
        stem = os.path.splitext(name)[0]

        if ext not in IMAGE_EXTS:
            continue

        if prefix is not None and not stem.startswith(prefix):
            continue

        files.append(os.path.join(image_dir, name))

    return sorted(files)


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

    x1 = int((xc - w / 2.0) * img_w)
    y1 = int((yc - h / 2.0) * img_h)
    x2 = int((xc + w / 2.0) * img_w)
    y2 = int((yc + h / 2.0) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return [x1, y1, x2, y2]


def xyxy_to_yolo(cls_id, x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(0, min(int(x2), img_w - 1))
    y2 = max(0, min(int(y2), img_h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1

    xc = (x1 + x2 + 1) / 2.0 / img_w
    yc = (y1 + y2 + 1) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h

    return [cls_id, xc, yc, w, h]


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1 + 1)
    inter_h = max(0, y2 - y1 + 1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0] + 1) * max(0, box1[3] - box1[1] + 1)
    area2 = max(0, box2[2] - box2[0] + 1) * max(0, box2[3] - box2[1] + 1)

    return inter / (area1 + area2 - inter + 1e-6)


def rotate_rgba(rgba, angle):
    h, w = rgba.shape[:2]
    center = (w / 2.0, h / 2.0)

    mat = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(mat[0, 0])
    sin = abs(mat[0, 1])

    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    mat[0, 2] += new_w / 2.0 - center[0]
    mat[1, 2] += new_h / 2.0 - center[1]

    rotated = cv2.warpAffine(
        rgba,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated


def resize_rgba(rgba, scale):
    h, w = rgba.shape[:2]

    new_w = max(2, int(w * scale))
    new_h = max(2, int(h * scale))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    return cv2.resize(rgba, (new_w, new_h), interpolation=interpolation)


def keep_largest_component(binary):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + int(np.argmax(areas))

    return np.where(labels == max_label, 255, 0).astype(np.uint8)


def clean_rgba_mask(rgba, alpha_thr=130, erode_iter=1, open_iter=1, feather=3):
    """
    清理 SAM 提取的 rgba:
    1. 去掉半透明背景
    2. 保留最大连通区域
    3. 轻微腐蚀，减少白边
    4. feather 平滑边缘
    """
    if rgba is None:
        return None

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return None

    out = rgba.copy()
    alpha = out[:, :, 3]

    binary = (alpha > alpha_thr).astype(np.uint8) * 255

    if binary.sum() == 0:
        return None

    binary = keep_largest_component(binary)

    kernel = np.ones((2, 2), np.uint8)

    if open_iter > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    if erode_iter > 0:
        binary = cv2.erode(binary, kernel, iterations=erode_iter)

    if binary.sum() == 0:
        return None

    alpha_f = binary.astype(np.float32) / 255.0

    if feather > 1:
        if feather % 2 == 0:
            feather += 1
        alpha_f = cv2.GaussianBlur(alpha_f, (feather, feather), 0)

    out[:, :, 3] = np.clip(alpha_f * 255, 0, 255).astype(np.uint8)

    return out


def crop_to_alpha_bbox(rgba, thr=10, pad=1):
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > thr)

    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = alpha.shape[:2]

    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w - 1, int(xs.max()) + pad)
    y2 = min(h - 1, int(ys.max()) + pad)

    return rgba[y1:y2 + 1, x1:x2 + 1].copy()


def lab_color_match(fg_bgr, alpha, bg_patch, strength=0.30):
    """
    LAB 颜色匹配。
    patch 背景和 SAM mask 都来自个人数据域，strength 不宜太高。
    建议 0.20~0.35。
    """
    mask = alpha > 0.2

    if mask.sum() < 10:
        return fg_bgr

    fg_lab = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_patch, cv2.COLOR_BGR2LAB).astype(np.float32)

    fg_pixels = fg_lab[mask]
    bg_pixels = bg_lab[mask]

    fg_mean = fg_pixels.mean(axis=0)
    fg_std = fg_pixels.std(axis=0) + 1e-6

    bg_mean = bg_pixels.mean(axis=0)
    bg_std = bg_pixels.std(axis=0) + 1e-6

    matched = (fg_lab - fg_mean) * (bg_std / fg_std) + bg_mean
    matched = np.clip(matched, 0, 255)

    mixed = fg_lab * (1.0 - strength) + matched * strength
    mixed = np.clip(mixed, 0, 255).astype(np.uint8)

    return cv2.cvtColor(mixed, cv2.COLOR_LAB2BGR)


def add_soft_shadow(bg_patch, alpha, strength=0.06, blur=3, dx=1, dy=1):
    h, w = alpha.shape[:2]

    shifted = np.zeros_like(alpha, dtype=np.float32)

    y1_src = max(0, -dy)
    y2_src = min(h, h - dy)
    x1_src = max(0, -dx)
    x2_src = min(w, w - dx)

    y1_dst = max(0, dy)
    y2_dst = min(h, h + dy)
    x1_dst = max(0, dx)
    x2_dst = min(w, w + dx)

    if y2_src <= y1_src or x2_src <= x1_src:
        return bg_patch

    shifted[y1_dst:y2_dst, x1_dst:x2_dst] = alpha[y1_src:y2_src, x1_src:x2_src]

    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        shifted = cv2.GaussianBlur(shifted, (blur, blur), 0)

    shadow = shifted * strength

    out = bg_patch.astype(np.float32)
    out = out * (1.0 - shadow[:, :, None])
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out


def paste_rgba_on_image(
    bg,
    rgba,
    x,
    y,
    feather=3,
    color_strength=0.30,
    use_poisson=True,
    use_shadow=True
):
    rgba = clean_rgba_mask(
        rgba,
        alpha_thr=130,
        erode_iter=1,
        open_iter=1,
        feather=feather
    )

    if rgba is None:
        return bg, None

    rgba = crop_to_alpha_bbox(rgba, thr=10, pad=1)

    if rgba is None:
        return bg, None

    img_h, img_w = bg.shape[:2]
    h, w = rgba.shape[:2]

    if w < 3 or h < 3:
        return bg, None

    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        return bg, None

    fg_bgr = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0

    if alpha.max() <= 0:
        return bg, None

    bg_patch = bg[y:y + h, x:x + w].copy()

    fg_bgr = lab_color_match(
        fg_bgr,
        alpha,
        bg_patch,
        strength=color_strength
    )

    if use_shadow:
        bg_patch_shadow = add_soft_shadow(
            bg_patch,
            alpha,
            strength=0.06,
            blur=3,
            dx=1,
            dy=1
        )
    else:
        bg_patch_shadow = bg_patch

    binary_mask = (alpha > 0.15).astype(np.uint8) * 255

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return bg, None

    bx1 = x + int(xs.min())
    by1 = y + int(ys.min())
    bx2 = x + int(xs.max())
    by2 = y + int(ys.max())

    bg_shadow = bg.copy()
    bg_shadow[y:y + h, x:x + w] = bg_patch_shadow

    if use_poisson:
        try:
            center = (x + w // 2, y + h // 2)
            cloned = cv2.seamlessClone(
                fg_bgr,
                bg_shadow,
                binary_mask,
                center,
                cv2.NORMAL_CLONE
            )
            return cloned, [bx1, by1, bx2, by2]
        except Exception:
            pass

    alpha_3 = alpha[:, :, None]
    blended = fg_bgr.astype(np.float32) * alpha_3 + bg_patch_shadow.astype(np.float32) * (1.0 - alpha_3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    out = bg.copy()
    out[y:y + h, x:x + w] = blended

    return out, [bx1, by1, bx2, by2]


def local_background_score(bg, x, y, w, h):
    patch = bg[y:y + h, x:x + w]

    if patch.size == 0:
        return 0.0

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    std = float(gray.std())

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())

    return std + edge_density * 50.0


def choose_position(
    bg,
    obj_w,
    obj_h,
    existing_boxes,
    max_iou=0.08,
    max_try=100,
    min_bg_score=6.0
):
    img_h, img_w = bg.shape[:2]

    if obj_w >= img_w or obj_h >= img_h:
        return None

    best = None
    best_score = -1.0

    for _ in range(max_try):
        x = random.randint(0, img_w - obj_w)
        y = random.randint(0, img_h - obj_h)

        candidate_box = [x, y, x + obj_w - 1, y + obj_h - 1]

        ok = True
        for eb in existing_boxes:
            if box_iou(candidate_box, eb) > max_iou:
                ok = False
                break

        if not ok:
            continue

        score = local_background_score(bg, x, y, obj_w, obj_h)

        if score > best_score:
            best = (x, y)
            best_score = score

        if score >= min_bg_score:
            return x, y

    return best


def copy_dataset(src_root, out_root, overwrite=False):
    if os.path.exists(out_root):
        if overwrite:
            print(f"删除已有输出目录: {out_root}")
            shutil.rmtree(out_root)
        else:
            raise FileExistsError(
                f"输出目录已存在: {out_root}\n"
                f"如需覆盖，请加 --overwrite"
            )

    ensure_dir(out_root)

    for split in ["train", "val", "test"]:
        src_split = os.path.join(src_root, split)

        if not os.path.exists(src_split):
            continue

        for sub in ["images", "labels"]:
            src_dir = os.path.join(src_split, sub)

            if not os.path.exists(src_dir):
                continue

            dst_dir = os.path.join(out_root, split, sub)
            ensure_dir(dst_dir)

            for name in tqdm(os.listdir(src_dir), desc=f"Copy {split}/{sub}"):
                src_file = os.path.join(src_dir, name)
                dst_file = os.path.join(dst_dir, name)

                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)

    src_yaml = os.path.join(src_root, "data.yaml")
    dst_yaml = os.path.join(out_root, "data.yaml")

    data = load_yaml(src_yaml)
    data["path"] = out_root
    data["train"] = "train/images"
    data["val"] = "val/images"

    if os.path.exists(os.path.join(out_root, "test", "images")):
        data["test"] = "test/images"
    else:
        data.pop("test", None)

    save_yaml(data, dst_yaml)


def save_image(path, img, jpg_quality=95):
    ext = os.path.splitext(path)[1].lower()

    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    else:
        cv2.imwrite(path, img)


def draw_preview(img, labels, out_path, target_class_id=30):
    vis = img.copy()
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

        color = (0, 0, 255) if int(cls_id) == int(target_class_id) else (0, 255, 0)

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_root",
        type=str,
        default="/root/ultralytics-8.3.27/mixed_dataset_patch1024",
        help="原始 mixed_dataset_patch1024 路径"
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default="/root/ultralytics-8.3.27/mixed_dataset_patch1024_sam_replace100",
        help="输出替换增强后的 mixed 数据集路径"
    )

    parser.add_argument(
        "--mask_rgba_dir",
        type=str,
        default="/root/ultralytics-8.3.27/hopper_sam_masks_datasets2/rgba",
        help="筛选后的稻飞虱 rgba mask 目录"
    )

    parser.add_argument(
        "--bg_prefix",
        type=str,
        default="patch_train",
        help="只选择该前缀开头的 train 图片作为增强背景"
    )

    parser.add_argument(
        "--target_class_id",
        type=int,
        default=30,
        help="稻飞虱在 mixed_dataset 中的类别 ID"
    )

    parser.add_argument(
        "--num_aug",
        type=int,
        default=100,
        help="随机选择多少张 patch_train 原图进行替换增强"
    )

    parser.add_argument("--paste_min", type=int, default=1)
    parser.add_argument("--paste_max", type=int, default=1)

    parser.add_argument("--scale_min", type=float, default=0.7)
    parser.add_argument("--scale_max", type=float, default=1.1)
    parser.add_argument("--rotate", type=float, default=10.0)

    parser.add_argument("--max_iou", type=float, default=0.08)
    parser.add_argument("--max_try", type=int, default=100)
    parser.add_argument("--min_bg_score", type=float, default=6.0)

    parser.add_argument("--color_strength", type=float, default=0.30)

    parser.add_argument("--preview_num", type=int, default=100)
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    src_train_img_dir = os.path.join(args.src_root, "train", "images")
    src_train_lab_dir = os.path.join(args.src_root, "train", "labels")

    if not os.path.exists(args.src_root):
        raise FileNotFoundError(f"找不到 src_root: {args.src_root}")

    if not os.path.exists(src_train_img_dir):
        raise FileNotFoundError(f"找不到 train/images: {src_train_img_dir}")

    if not os.path.exists(src_train_lab_dir):
        raise FileNotFoundError(f"找不到 train/labels: {src_train_lab_dir}")

    if not os.path.exists(args.mask_rgba_dir):
        raise FileNotFoundError(f"找不到 mask_rgba_dir: {args.mask_rgba_dir}")

    rgba_files = sorted([
        os.path.join(args.mask_rgba_dir, f)
        for f in os.listdir(args.mask_rgba_dir)
        if f.lower().endswith(".png")
    ])

    bg_images = find_images(src_train_img_dir, prefix=args.bg_prefix)

    if len(rgba_files) == 0:
        raise RuntimeError(f"mask_rgba_dir 中没有 PNG: {args.mask_rgba_dir}")

    if len(bg_images) == 0:
        raise RuntimeError(
            f"没有找到以 {args.bg_prefix} 开头的背景图，请检查 train/images 文件名"
        )

    selected_bg_images = random.sample(bg_images, min(args.num_aug, len(bg_images)))

    print("\n========== 当前配置 ==========")
    print(f"src_root        : {args.src_root}")
    print(f"out_root        : {args.out_root}")
    print(f"mask_rgba_dir   : {args.mask_rgba_dir}")
    print(f"bg_prefix       : {args.bg_prefix}")
    print(f"target_class_id : {args.target_class_id}")
    print(f"rgba 数量       : {len(rgba_files)}")
    print(f"个人 patch 背景 : {len(bg_images)}")
    print(f"计划替换增强数量 : {args.num_aug}")
    print(f"实际选择背景数量 : {len(selected_bg_images)}")
    print(f"paste           : {args.paste_min} ~ {args.paste_max}")
    print(f"scale           : {args.scale_min} ~ {args.scale_max}")
    print(f"rotate          : ±{args.rotate}")
    print(f"color_strength  : {args.color_strength}")
    print("增强模式        : 替换模式，不额外新增图片")

    print("\n========== 复制 mixed_dataset_patch1024 ==========")
    copy_dataset(
        src_root=args.src_root,
        out_root=args.out_root,
        overwrite=args.overwrite
    )

    out_train_img_dir = os.path.join(args.out_root, "train", "images")
    out_train_lab_dir = os.path.join(args.out_root, "train", "labels")
    preview_dir = os.path.join(args.out_root, "preview_samhopper_replace")

    ensure_dir(out_train_img_dir)
    ensure_dir(out_train_lab_dir)
    ensure_dir(preview_dir)

    generated = 0
    failed = 0
    records = []

    for aug_idx, bg_path in enumerate(tqdm(selected_bg_images, desc="Replace selected patch_train images with SAM aug")):
        bg = cv2.imread(bg_path)

        if bg is None:
            failed += 1
            records.append([bg_path, "failed", "image_read_failed"])
            continue

        img_h, img_w = bg.shape[:2]
        stem = image_stem(bg_path)

        src_label_path = os.path.join(src_train_lab_dir, stem + ".txt")
        labels = read_yolo_labels(src_label_path)

        existing_boxes = [yolo_to_xyxy(lab, img_w, img_h) for lab in labels]

        new_img = bg.copy()
        new_labels = labels.copy()

        paste_num = random.randint(args.paste_min, args.paste_max)
        pasted_count = 0

        for _ in range(paste_num):
            rgba_path = random.choice(rgba_files)
            rgba = cv2.imread(rgba_path, cv2.IMREAD_UNCHANGED)

            if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
                continue

            scale = random.uniform(args.scale_min, args.scale_max)
            angle = random.uniform(-args.rotate, args.rotate)

            rgba_aug = resize_rgba(rgba, scale)
            rgba_aug = rotate_rgba(rgba_aug, angle)

            rgba_clean = clean_rgba_mask(
                rgba_aug,
                alpha_thr=130,
                erode_iter=1,
                open_iter=1,
                feather=3
            )

            if rgba_clean is None:
                continue

            rgba_clean = crop_to_alpha_bbox(rgba_clean, thr=10, pad=1)

            if rgba_clean is None:
                continue

            rh, rw = rgba_clean.shape[:2]

            if rw < 3 or rh < 3:
                continue

            if rw >= img_w or rh >= img_h:
                continue

            pos = choose_position(
                new_img,
                rw,
                rh,
                existing_boxes,
                max_iou=args.max_iou,
                max_try=args.max_try,
                min_bg_score=args.min_bg_score
            )

            if pos is None:
                continue

            x, y = pos

            pasted_img, new_box = paste_rgba_on_image(
                new_img,
                rgba_clean,
                x,
                y,
                feather=1,
                color_strength=args.color_strength,
                use_poisson=False,
                use_shadow=True
            )

            if new_box is None:
                continue

            bx1, by1, bx2, by2 = new_box

            if bx2 - bx1 + 1 < 3 or by2 - by1 + 1 < 3:
                continue

            new_label = xyxy_to_yolo(
                args.target_class_id,
                bx1,
                by1,
                bx2,
                by2,
                img_w,
                img_h
            )

            if new_label is None:
                continue

            new_img = pasted_img
            new_labels.append(new_label)
            existing_boxes.append(new_box)
            pasted_count += 1

        if pasted_count == 0:
            failed += 1
            records.append([bg_path, "failed", "paste_failed"])
            continue

        # 替换模式：保持原图文件名，覆盖新数据集中的对应图片和标签
        orig_img_name = os.path.basename(bg_path)
        out_img_path = os.path.join(out_train_img_dir, orig_img_name)
        out_lab_path = os.path.join(out_train_lab_dir, stem + ".txt")

        save_image(out_img_path, new_img, jpg_quality=args.jpg_quality)
        write_yolo_labels(out_lab_path, new_labels)

        if generated < args.preview_num:
            preview_path = os.path.join(
                preview_dir,
                stem + "_replace_preview.jpg"
            )
            draw_preview(
                new_img,
                new_labels,
                preview_path,
                target_class_id=args.target_class_id
            )

        generated += 1
        records.append([bg_path, "success", f"pasted_{pasted_count}"])

    csv_path = os.path.join(args.out_root, "sam_replace_records.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_image", "status", "note"])
        writer.writerows(records)

    print("\n========== 完成 ==========")
    print(f"成功替换增强图片 : {generated}")
    print(f"失败/跳过数量    : {failed}")
    print(f"新数据集目录      : {args.out_root}")
    print(f"预览图目录        : {preview_dir}")
    print(f"替换记录 CSV      : {csv_path}")
    print(f"训练 data.yaml    : {os.path.join(args.out_root, 'data.yaml')}")


if __name__ == "__main__":
    main()