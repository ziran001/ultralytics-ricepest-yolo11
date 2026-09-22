"""Compare YOLO11 and FRACAL at fixed/matched recall and trace missed objects.

One validation inference pass feeds both score variants into identical NMS. This
script is a diagnostic of IoU=0.5 detection, not a replacement for COCO mAP.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from fracal_yolo11 import FracalCalibrator, dataset_config


FOCUS_NAMES = ("Chilo suppressalis", "Cnaphalocrocis medinalis", "Rice planthopper")
TARGET_RECALL = {"Chilo suppressalis": 0.8849, "Cnaphalocrocis medinalis": 0.9112, "Rice planthopper": 0.9215}


def pairwise_iou(boxes_a, boxes_b):
    """Return pairwise IoU for two xyxy numpy arrays."""
    if not len(boxes_a) or not len(boxes_b):
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(np.maximum(boxes_a[:, 2:] - boxes_a[:, :2], 0), axis=1)
    area_b = np.prod(np.maximum(boxes_b[:, 2:] - boxes_b[:, :2], 0), axis=1)
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def score_match(gt_boxes, detections, iou_threshold=0.5):
    """Greedily assign score-sorted same-class detections to unused GT boxes.

    Returns a score for every GT (zero if missed), plus a TP flag for each
    detection. Prefixes of this ordering represent every confidence threshold.
    """
    gt_scores = np.zeros(len(gt_boxes), dtype=np.float32)
    tp = np.zeros(len(detections), dtype=bool)
    if not len(gt_boxes) or not len(detections):
        return gt_scores, tp
    ious = pairwise_iou(gt_boxes, detections[:, :4])
    for j in np.argsort(-detections[:, 4], kind="stable"):
        candidates = ious[:, j].copy()
        candidates[gt_scores > 0] = -1
        i = int(candidates.argmax())
        if candidates[i] >= iou_threshold:
            tp[j] = True
            gt_scores[i] = detections[j, 4]
    return gt_scores, tp


def curve(rows, total_gt):
    """Compute raw confidence, precision, recall arrays from (score, TP) rows."""
    if not rows:
        return {"score": np.empty(0), "precision": np.empty(0), "recall": np.empty(0)}
    values = np.asarray(rows, dtype=np.float64)
    order = np.argsort(-values[:, 0], kind="stable")
    scores = values[order, 0]
    tp = np.cumsum(values[order, 1])
    return {"score": scores, "precision": tp / np.arange(1, len(tp) + 1), "recall": tp / total_gt}


def at_threshold(pr, threshold):
    """Metrics at a fixed confidence threshold, inclusive of equal scores."""
    kept = int(np.count_nonzero(pr["score"] >= threshold))
    if not kept:
        return {"threshold": float(threshold), "P": 0.0, "R": 0.0, "detections": 0}
    return {"threshold": float(threshold), "P": float(pr["precision"][kept - 1]),
            "R": float(pr["recall"][kept - 1]), "detections": kept}


def at_recall(pr, target):
    """Highest threshold reaching target recall; report unattainable targets."""
    indices = np.flatnonzero(pr["recall"] >= target)
    if not len(indices):
        return {"target_R": float(target), "reachable": False,
                "max_R": float(pr["recall"][-1]) if len(pr["score"]) else 0.0}
    index = int(indices[0])
    result = at_threshold(pr, pr["score"][index])
    result.update(target_R=float(target), reachable=True)
    return result


def stage(matched_score, post_score, raw_score, other_score, threshold):
    """Assign mutually exclusive, observable reasons for one GT miss."""
    if matched_score >= threshold:
        return "recovered"
    if post_score >= threshold:
        return "matching_conflict"
    if raw_score >= threshold:
        return "nms_or_limit"
    if raw_score > 0:
        return "score_below_threshold"
    if other_score > 0:
        return "other_class_only"
    return "no_iou50_raw_box"


def analyze(records, focus_ids, names, floor, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {"iou_match": 0.5, "collection_conf": floor, "classes": {}}
    case_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)
    for axis, class_id in zip(axes, focus_ids):
        name = names[class_id]
        class_records = [entry for entry in records if entry["class_id"] == class_id]
        n_gt = sum(len(entry["gt"]) for entry in class_records)
        result = {"gt_count": n_gt, "target_recall": TARGET_RECALL[name], "arms": {}}
        curves = {}
        for arm in ("baseline", "fracal"):
            rows = []
            for entry in class_records:
                pred = entry[arm]["pred"]
                matched, tp = score_match(entry["gt"], pred)
                entry[arm]["matched"] = matched
                rows.extend(zip(pred[:, 4].tolist(), tp.tolist()))
            pr = curve(rows, n_gt)
            curves[arm] = pr
            target = at_recall(pr, TARGET_RECALL[name])
            # Choose the highest score at which the requested recall is obtained.
            threshold = target["threshold"] if target["reachable"] else floor
            counts = {}
            for entry in class_records:
                detail = entry[arm]
                for index in range(len(entry["gt"])):
                    label = stage(detail["matched"][index], detail["post_max"][index],
                                  detail["raw_max"][index], detail["other_raw_max"][index], threshold)
                    counts[label] = counts.get(label, 0) + 1
                    if label != "recovered":
                        case_rows.append({"image": entry["image"], "class": name, "arm": arm,
                                          "gt_index_in_class": index, "stage": label,
                                          "threshold": threshold,
                                          "matched_score": float(detail["matched"][index]),
                                          "post_max_score": float(detail["post_max"][index]),
                                          "raw_max_score": float(detail["raw_max"][index]),
                                          "other_class_raw_max_score": float(detail["other_raw_max"][index])})
            result["arms"][arm] = {"at_0.25": at_threshold(pr, 0.25),
                                    "at_0.01": at_threshold(pr, 0.01),
                                    "at_0.001": at_threshold(pr, 0.001),
                                    "at_target_recall": target,
                                    "stage_counts_at_target_recall": counts,
                                    "max_recall_at_collection_floor": at_threshold(pr, floor)["R"]}
            axis.plot(pr["recall"], pr["precision"], label=arm, linewidth=1.5)
        results["classes"][name] = result
        axis.set_title(name, fontsize=10)
        axis.set_xlabel("Recall @ IoU 0.5")
        axis.grid(alpha=0.3)
        axis.axvline(TARGET_RECALL[name], color="gray", linestyle="--", linewidth=0.8)
        axis.legend()
    axes[0].set_ylabel("Precision")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output / "focus_pr_iou50.png", dpi=170)
    plt.close(fig)
    (output / "recall_analysis.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "missed_cases.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0]) if case_rows else
                                ["image", "class", "arm", "gt_index_in_class", "stage", "threshold",
                                 "matched_score", "post_max_score", "raw_max_score", "other_class_raw_max_score"])
        writer.writeheader()
        writer.writerows(case_rows)
    return results


def run(args):
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import box_iou

    with Path(args.stats).open(encoding="utf-8") as stream:
        stats = json.load(stream)
    _, _, names = dataset_config(args.data)
    model = YOLO(args.weights)
    if stats["names"] != names or [model.names[i] for i in range(len(names))] != names:
        raise ValueError("Dataset, statistics, and checkpoint class orders differ")
    focus_ids = [names.index(name) for name in FOCUS_NAMES]
    calibrator = FracalCalibrator(stats, args.beta, args.fractal_lambda)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []

    class TraceValidator(DetectionValidator):
        def postprocess(self, preds):
            raw = preds[0] if isinstance(preds, (tuple, list)) else preds
            calibrated = calibrator(raw)
            self.raw = raw
            self.calibrated = calibrated
            common = dict(conf_thres=self.args.conf, iou_thres=self.args.iou,
                          multi_label=True, agnostic=False, max_det=self.args.max_det,
                          max_time_img=10.0, in_place=False)
            self.fracal_detections = ops.non_max_suppression(calibrated, **common)
            return ops.non_max_suppression(raw, **common)

        def update_metrics(self, preds, batch):
            for si, baseline in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                gt_class = pbatch["cls"].detach().cpu().numpy().astype(int)
                gt_boxes = pbatch["bbox"].detach().cpu().numpy()
                image_shape = batch["img"].shape[2:]
                raw_boxes = ops.xywh2xyxy(self.raw[si, :4, :].T.float())
                original_scores = self.raw[si, 4:, :].float()
                calibrated_scores = self.calibrated[si, 4:, :].float()
                prepared = {
                    "baseline": self._prepare_pred(baseline, pbatch).detach().cpu().numpy(),
                    "fracal": self._prepare_pred(self.fracal_detections[si], pbatch).detach().cpu().numpy(),
                }
                for class_id in focus_ids:
                    local_gt = gt_boxes[gt_class == class_id]
                    if len(local_gt):
                        gt_input = ops.xywh2xyxy(batch["bboxes"][batch["batch_idx"] == si][
                            torch.as_tensor(gt_class == class_id, device=self.device)])
                        gt_input *= torch.tensor(image_shape, device=self.device)[[1, 0, 1, 0]]
                        overlaps = box_iou(gt_input.float(), raw_boxes) >= 0.5
                    entry = {"image": str(batch["im_file"][si]), "class_id": class_id, "gt": local_gt}
                    for arm, scores in (("baseline", original_scores), ("fracal", calibrated_scores)):
                        selected = prepared[arm]
                        selected = selected[selected[:, 5] == class_id]
                        if len(selected):
                            post_iou = pairwise_iou(local_gt, selected[:, :4])
                            post_max = np.where(post_iou >= 0.5, selected[None, :, 4], 0).max(axis=1)
                        else:
                            post_max = np.zeros(len(local_gt), dtype=np.float32)
                        if len(local_gt):
                            same = torch.where(overlaps, scores[class_id][None, :], 0).amax(dim=1)
                            # Multi-label NMS has no argmax; this is diagnostic evidence only.
                            other = scores.clone()
                            other[class_id] = 0
                            other = torch.where(overlaps, other.amax(dim=0)[None, :], 0).amax(dim=1)
                            same = same.detach().cpu().numpy()
                            other = other.detach().cpu().numpy()
                        else:
                            same = np.empty(0, dtype=np.float32)
                            other = np.empty(0, dtype=np.float32)
                        entry[arm] = {"pred": selected, "post_max": post_max,
                                      "raw_max": same, "other_raw_max": other}
                    records.append(entry)
            super().update_metrics(preds, batch)

    model.val(validator=TraceValidator, data=args.data, split="val", imgsz=args.imgsz,
              batch=args.batch, device=args.device, workers=args.workers,
              conf=args.conf, iou=args.iou, max_det=args.max_det,
              plots=False, project=str(output), name="validator", exist_ok=True, verbose=False)
    result = analyze(records, focus_ids, names, args.conf, output)
    result["inputs"] = {"data": str(Path(args.data).resolve()), "weights": str(Path(args.weights).resolve()),
                        "stats": str(Path(args.stats).resolve()), "imgsz": args.imgsz, "batch": args.batch,
                        "workers": args.workers, "nms_iou": args.iou, "max_det": args.max_det,
                        "beta": args.beta, "fractal_lambda": args.fractal_lambda}
    for class_id in focus_ids:
        class_result = result["classes"][names[class_id]]
        original = class_result["arms"]["baseline"]["at_target_recall"]
        if original["reachable"]:
            p = original["threshold"]
            offset = float(calibrator.offsets[class_id])
            mapped = p / (1 + ((1 - p) / p) * math.exp(-offset))
            class_result["fracal_equivalent_of_baseline_threshold"] = mapped
    (output / "recall_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved {output / 'recall_analysis.json'}, {output / 'focus_pr_iou50.png'}, and {output / 'missed_cases.csv'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.00001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--fractal-lambda", type=float, default=2.0)
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
