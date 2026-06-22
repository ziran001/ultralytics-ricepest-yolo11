import os
import cv2
import yaml
import shutil
import random
import argparse
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_yaml(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", None)
    if names is None:
        raise ValueError(f"data.yaml 中没有 names 字段: {data_yaml}")

    if isinstance(names, list):
        id_to_name = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        id_to_name = {int(k): v for k, v in names.items()}
    else:
        raise ValueError("data.yaml 中 names 格式错误，应为 list 或 dict")

    return data, id_to_name


def save_yaml(data, out_yaml):
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def find_images(image_dir):
    files = []
    if not os.path.exists(image_dir):
        return files

    for name in os.listdir(image_dir):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
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

    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

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
    V2：清理 SAM mask。
    1. 去掉半透明边缘
    2. 保留最大连通区域
    3. 轻微开运算
    4. 轻微腐蚀，减少白边
    5. feather 平滑边缘
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


def lab_color_match(fg_bgr, alpha, bg_patch, strength=0.35):
    """
    V2：LAB 颜色匹配。
    strength=0.35 是 V2 的默认值，融合较自然，但在浅色背景上可能略偏白。
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


def add_soft_shadow(bg_patch, alpha, strength=0.08, blur=3, dx=1, dy=1):
    """
    V2：添加极轻微阴影，减少漂浮感。
    """
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
    color_match=True,
    color_strength=0.35,
    use_poisson=True,
    use_shadow=True
):
    """
    V2 粘贴流程：
    clean mask -> crop -> LAB 颜色匹配 -> 轻微阴影 -> Poisson seamlessClone
    """
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

    if color_match:
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
            strength=0.08,
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
    """
    背景评分，避免贴到过于干净、过于平坦的区域。
    """
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
    min_bg_score=8.0
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


def copy_dataset_to_new(src_root, dst_root, overwrite=False):
    """
    复制 mixed_dataset 到新目录，不修改原始数据集。
    """
    if os.path.exists(dst_root):
        if overwrite:
            print(f"删除已存在输出目录: {dst_root}")
            shutil.rmtree(dst_root)
        else:
            raise FileExistsError(
                f"输出目录已存在: {dst_root}\n"
                f"需要覆盖时请加 --overwrite"
            )

    ensure_dir(dst_root)

    for split in ["train", "val", "test"]:
        src_split = os.path.join(src_root, split)

        if not os.path.exists(src_split):
            continue

        for sub in ["images", "labels"]:
            src_dir = os.path.join(src_split, sub)

            if not os.path.exists(src_dir):
                continue

            dst_dir = os.path.join(dst_root, split, sub)
            ensure_dir(dst_dir)

            for name in tqdm(os.listdir(src_dir), desc=f"Copy {split}/{sub}"):
                src_file = os.path.join(src_dir, name)
                dst_file = os.path.join(dst_dir, name)

                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)

    src_yaml = os.path.join(src_root, "data.yaml")
    dst_yaml = os.path.join(dst_root, "data.yaml")

    data, _ = load_yaml(src_yaml)

    data["path"] = dst_root
    data["train"] = "train/images"

    if os.path.exists(os.path.join(dst_root, "val", "images")):
        data["val"] = "val/images"

    if os.path.exists(os.path.join(dst_root, "test", "images")):
        data["test"] = "test/images"
    else:
        data.pop("test", None)

    save_yaml(data, dst_yaml)


def draw_preview(img, labels, out_path, target_class_id=30):
    """
    preview_samhopper：
    红框：类别 30，也就是新加稻飞虱
    绿框：其他已有目标
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    for cls_id, xc, yc, bw, bh in labels:
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        color = (0, 0, 255) if int(cls_id) == int(target_class_id) else (0, 255, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            vis,
            str(int(cls_id)),
            (x1, max(0, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )

    cv2.imwrite(out_path, vis)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixed_root",
        type=str,
        default="/root/ultralytics-8.3.27/mixed_dataset 1+1",
        help="原始混合数据集路径"
    )

    parser.add_argument(
        "--mask_rgba_dir",
        type=str,
        default="/root/ultralytics-8.3.27/hopper_sam_masks_datasets2/rgba",
        help="筛选后的稻飞虱 RGBA 目录"
    )

    parser.add_argument(
        "--out_root",
        type=str,
        default="/root/ultralytics-8.3.27/mixed_dataset_1+1_sam_aug_test50_goodrgba_v2",
        help="输出增强后的新数据集路径"
    )

    parser.add_argument(
        "--target_class_id",
        type=int,
        default=30,
        help="混合数据集中的稻飞虱类别 ID，必须是 30"
    )

    parser.add_argument(
        "--num_aug",
        type=int,
        default=50,
        help="生成多少张增强图"
    )

    parser.add_argument(
        "--paste_min",
        type=int,
        default=1,
        help="每张图最少贴几个稻飞虱"
    )

    parser.add_argument(
        "--paste_max",
        type=int,
        default=1,
        help="每张图最多贴几个稻飞虱"
    )

    parser.add_argument(
        "--scale_min",
        type=float,
        default=0.6,
        help="最小缩放比例"
    )

    parser.add_argument(
        "--scale_max",
        type=float,
        default=1.0,
        help="最大缩放比例"
    )

    parser.add_argument(
        "--rotate",
        type=float,
        default=10.0,
        help="随机旋转角度范围，例如 10 表示 -10 到 +10"
    )

    parser.add_argument(
        "--max_iou",
        type=float,
        default=0.08,
        help="与原有标注框最大允许 IoU，避免遮挡原目标"
    )

    parser.add_argument(
        "--max_try",
        type=int,
        default=100,
        help="每个目标尝试放置次数"
    )

    parser.add_argument(
        "--min_bg_score",
        type=float,
        default=8.0,
        help="背景纹理最低分"
    )

    parser.add_argument(
        "--color_strength",
        type=float,
        default=0.35,
        help="LAB 颜色匹配强度，V2 默认 0.35"
    )

    parser.add_argument(
        "--preview_num",
        type=int,
        default=50,
        help="保存多少张带框预览图"
    )

    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=95
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="输出目录存在时删除后重建"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    mixed_root = args.mixed_root
    mask_rgba_dir = args.mask_rgba_dir
    out_root = args.out_root

    train_img_dir = os.path.join(mixed_root, "train", "images")
    train_lab_dir = os.path.join(mixed_root, "train", "labels")
    data_yaml = os.path.join(mixed_root, "data.yaml")

    if not os.path.exists(mixed_root):
        raise FileNotFoundError(f"找不到 mixed_root: {mixed_root}")

    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"找不到 train/images: {train_img_dir}")

    if not os.path.exists(train_lab_dir):
        raise FileNotFoundError(f"找不到 train/labels: {train_lab_dir}")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"找不到 data.yaml: {data_yaml}")

    if not os.path.exists(mask_rgba_dir):
        raise FileNotFoundError(f"找不到 mask_rgba_dir: {mask_rgba_dir}")

    data, id_to_name = load_yaml(data_yaml)

    print("\n========== mixed_dataset 类别列表 ==========")
    for cid, cname in id_to_name.items():
        print(f"{cid}: {cname}")

    if args.target_class_id not in id_to_name:
        raise ValueError(
            f"target_class_id={args.target_class_id} 不在 mixed_dataset 的 data.yaml 中"
        )

    rgba_files = sorted([
        os.path.join(mask_rgba_dir, f)
        for f in os.listdir(mask_rgba_dir)
        if f.lower().endswith(".png")
    ])

    bg_images = find_images(train_img_dir)

    if len(rgba_files) == 0:
        raise RuntimeError(f"RGBA 目录没有 PNG 文件: {mask_rgba_dir}")

    if len(bg_images) == 0:
        raise RuntimeError(f"train/images 没有图片: {train_img_dir}")

    print("\n========== 当前配置 ==========")
    print(f"mixed_root      : {mixed_root}")
    print(f"mask_rgba_dir   : {mask_rgba_dir}")
    print(f"out_root        : {out_root}")
    print(f"target_class_id : {args.target_class_id}")
    print(f"target_class    : {id_to_name[args.target_class_id]}")
    print(f"rgba 数量       : {len(rgba_files)}")
    print(f"训练背景图数量  : {len(bg_images)}")
    print(f"num_aug         : {args.num_aug}")
    print(f"paste           : {args.paste_min} ~ {args.paste_max}")
    print(f"scale           : {args.scale_min} ~ {args.scale_max}")
    print(f"rotate          : ±{args.rotate}")
    print(f"color_strength  : {args.color_strength}")

    print("\n========== 复制原 mixed_dataset 到新目录 ==========")
    copy_dataset_to_new(mixed_root, out_root, overwrite=args.overwrite)

    out_train_img_dir = os.path.join(out_root, "train", "images")
    out_train_lab_dir = os.path.join(out_root, "train", "labels")
    preview_dir = os.path.join(out_root, "preview_samhopper")

    ensure_dir(out_train_img_dir)
    ensure_dir(out_train_lab_dir)
    ensure_dir(preview_dir)

    generated = 0
    failed = 0

    for aug_idx in tqdm(range(args.num_aug), desc="Generate SAM hopper copy-paste V2"):
        bg_path = random.choice(bg_images)
        bg = cv2.imread(bg_path)

        if bg is None:
            failed += 1
            continue

        img_h, img_w = bg.shape[:2]

        stem = os.path.splitext(os.path.basename(bg_path))[0]
        label_path = os.path.join(train_lab_dir, stem + ".txt")

        labels = read_yolo_labels(label_path)
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
                feather=3,
                color_match=True,
                color_strength=args.color_strength,
                use_poisson=True,
                use_shadow=True
            )

            if new_box is None:
                continue

            bx1, by1, bx2, by2 = new_box
            bw = bx2 - bx1 + 1
            bh = by2 - by1 + 1

            if bw < 3 or bh < 3:
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
            continue

        out_name = f"{stem}_samhopper_v2_aug_{aug_idx:05d}.jpg"
        out_img_path = os.path.join(out_train_img_dir, out_name)
        out_lab_path = os.path.join(out_train_lab_dir, os.path.splitext(out_name)[0] + ".txt")

        cv2.imwrite(
            out_img_path,
            new_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality]
        )

        write_yolo_labels(out_lab_path, new_labels)

        if generated < args.preview_num:
            preview_path = os.path.join(
                preview_dir,
                os.path.splitext(out_name)[0] + "_preview.jpg"
            )
            draw_preview(
                new_img,
                new_labels,
                preview_path,
                target_class_id=args.target_class_id
            )

        generated += 1

    print("\n========== 完成 ==========")
    print(f"成功生成增强图片 : {generated}")
    print(f"失败/跳过数量    : {failed}")
    print(f"新数据集目录      : {out_root}")
    print(f"增强图目录        : {out_train_img_dir}")
    print(f"增强标签目录      : {out_train_lab_dir}")
    print(f"预览图目录        : {preview_dir}")
    print(f"训练 data.yaml    : {os.path.join(out_root, 'data.yaml')}")


if __name__ == "__main__":
    main()