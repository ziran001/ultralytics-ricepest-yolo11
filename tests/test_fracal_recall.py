"""Small deterministic checks for fixed-threshold FRACAL diagnostics."""

import numpy as np

from analyze_fracal_recall import at_recall, at_threshold, curve, score_match, stage


def test_score_match_recomputes_at_each_threshold():
    gt = np.array([[0, 0, 10, 10]], dtype=np.float32)
    detections = np.array([[0, 0, 10, 10, 0.8], [0, 0, 10, 10, 0.2]], dtype=np.float32)
    scores, tp = score_match(gt, detections)
    np.testing.assert_array_equal(tp, [True, False])
    np.testing.assert_allclose(scores, [0.8])
    pr = curve(list(zip(detections[:, 4], tp)), 1)
    assert at_threshold(pr, 0.5)["P"] == 1
    assert abs(at_recall(pr, 1)["threshold"] - 0.8) < 1e-6


def test_empty_ground_truth_predictions_are_false_positives():
    scores, tp = score_match(np.empty((0, 4)), np.array([[0, 0, 10, 10, 0.9]]))
    assert not len(scores)
    assert not tp.any()
    pr = curve([(0.9, False), (0.8, True)], 1)
    assert at_threshold(pr, 0.5)["P"] == 0.5
    assert at_threshold(pr, 0.5)["R"] == 1


def test_miss_stages_are_ordered():
    assert stage(0.6, 0.6, 0.6, 0, 0.5) == "recovered"
    assert stage(0, 0.6, 0.6, 0, 0.5) == "matching_conflict"
    assert stage(0, 0, 0.6, 0, 0.5) == "nms_or_limit"
    assert stage(0, 0, 0.2, 0, 0.5) == "score_below_threshold"
    assert stage(0, 0, 0, 0.4, 0.5) == "other_class_only"
    assert stage(0, 0, 0, 0, 0.5) == "no_iou50_raw_box"
