"""Checks for FRACAL statistics and calibration offsets without model weights."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fracal_yolo11 import box_dimension, calibration_shift, collect_statistics


class TestFracalYolo11(unittest.TestCase):
    def test_weighted_manifest_repeats_affect_frequency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for name, cls in [("a", 0), ("b", 1)]:
                (image_dir / f"{name}.jpg").write_bytes(b"image")
                (label_dir / f"{name}.txt").write_text(f"{cls} 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (root / "train.txt").write_text(
                "./images/train/a.jpg\n./images/train/a.jpg\n./images/train/b.jpg\n", encoding="utf-8"
            )
            data = root / "data.yaml"
            data.write_text(f"path: {root.as_posix()}\ntrain: train.txt\nnames: [a, b]\n", encoding="utf-8")
            stats = collect_statistics(data)
            self.assertEqual(stats["num_train_entries"], 3)
            self.assertEqual(stats["instance_counts"], [2, 1])
            self.assertEqual(stats["fractal_dimensions"], [1.0, 1.0])
            json.dumps(stats)

    def test_box_dimension_and_zero_offset(self):
        self.assertEqual(box_dimension([(0.5, 0.5)] * 3), 1.0)
        stats = {"instance_counts": [10, 10], "fractal_dimensions": [1.0, 1.0]}
        np.testing.assert_allclose(calibration_shift(stats), [0.0, 0.0], atol=1e-12)

    def test_rare_class_receives_larger_offset(self):
        stats = {"instance_counts": [100, 10], "fractal_dimensions": [2.0, 1.0]}
        offsets = calibration_shift(stats, beta=10.0, fractal_lambda=2.0)
        self.assertGreater(offsets[1], offsets[0])
        with self.assertRaises(ValueError):
            calibration_shift(stats, beta=1.0)


if __name__ == "__main__":
    unittest.main()
