"""Focused tests for the optional EFL-family YOLO detection classification losses."""

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import EqualizedFocalLoss
from ultralytics.utils.metrics import compute_ap


class TestEqualizedFocalLoss(unittest.TestCase):
    def test_compute_ap_supports_current_numpy(self):
        ap, _, _ = compute_ap(np.array([0.0, 0.5, 1.0]), np.array([1.0, 0.75, 0.5]))
        self.assertTrue(np.isfinite(ap))

    def test_model_yaml_selects_eqfl_without_architecture_change(self):
        model = YOLO("ultralytics/cfg/models/11/yolo11n-eqfl.yaml")
        vanilla = DetectionModel("ultralytics/cfg/models/11/yolo11n.yaml", nc=32, verbose=False)
        self.assertEqual(model.overrides["cls_loss"], "eqfl")
        self.assertEqual(model.overrides["efl_gamma"], 2.0)
        self.assertEqual(model.model.yaml["scale"], "n")
        self.assertEqual(model.model.yaml["nc"], 32)
        self.assertEqual(
            {key: tuple(value.shape) for key, value in model.model.state_dict().items()},
            {key: tuple(value.shape) for key, value in vanilla.state_dict().items()},
        )
        self.assertEqual(model.model.args["cls_loss"], "eqfl")
        model.model.args = SimpleNamespace(**model.model.args)  # DetectionTrainer supplies this before loss creation.
        self.assertEqual(model.model.init_criterion().cls_loss_name, "eqfl")

    def test_eqfl_starts_as_fixed_quality_focal_loss(self):
        logits = torch.tensor([[[0.2, -0.4]]], requires_grad=True)
        quality = torch.tensor([[[0.7, 0.0]]])
        loss = EqualizedFocalLoss(2, mode="eqfl")
        expected = F.binary_cross_entropy_with_logits(logits, quality, reduction="none")
        expected = expected * (quality - logits.sigmoid()).abs().square()
        torch.testing.assert_close(loss(logits, quality), expected)
        loss(logits, quality).sum().backward()
        self.assertGreater(loss.pos_grad[0].item(), 0)
        self.assertGreater(loss.neg_grad[1].item(), 0)

    def test_adaptive_gamma_and_state_restore(self):
        loss = EqualizedFocalLoss(2, mode="eqfl")
        logits = torch.zeros(1, 1, 2, requires_grad=True)
        quality = torch.tensor([[[0.8, 0.0]]])
        loss(logits, quality).sum().backward()
        self.assertGreater(loss.neg_grad[1].item(), 0)
        self.assertEqual(loss.pos_grad[1].item(), 0)
        resumed = EqualizedFocalLoss(2, mode="eqfl")
        resumed.load_state_dict(loss.state_dict())
        torch.testing.assert_close(resumed(logits.detach(), quality), loss(logits.detach(), quality))
        fixed = EqualizedFocalLoss(2, mode="qfl")
        self.assertNotEqual(
            resumed(logits.detach(), quality)[0, 0, 1].item(),
            fixed(logits.detach(), quality)[0, 0, 1].item(),
        )

    def test_hard_label_efl_formula(self):
        logits = torch.tensor([[[0.4, -0.2]]])
        quality = torch.tensor([[[0.6, 0.0]]])
        loss = EqualizedFocalLoss(2, mode="efl")
        binary = (quality > 0).float()
        probability = logits.sigmoid()
        pt = torch.where(binary.bool(), probability, 1 - probability)
        alpha = torch.where(binary.bool(), 0.25, 0.75)
        expected = F.binary_cross_entropy_with_logits(logits, binary, reduction="none") * (1 - pt).square() * alpha
        torch.testing.assert_close(loss(logits, quality), expected)

    def test_overflow_does_not_poison_adaptive_state(self):
        loss = EqualizedFocalLoss(2, mode="eqfl")
        positive = torch.tensor([[[True, False]]])
        loss._collect_grad(torch.tensor([[[float("inf"), 1.0]]]), positive)
        self.assertEqual(loss.pos_grad.sum().item(), 0)
        self.assertEqual(loss.neg_grad.sum().item(), 0)

    def test_detection_forward_and_state(self):
        model = DetectionModel("ultralytics/cfg/models/11/yolo11n.yaml", nc=32, verbose=False).train()
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, cls_loss="eqfl", efl_gamma=2.0, efl_scale=4.0)
        batch = {
            "img": torch.rand(1, 3, 64, 64),
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        }
        total, items = model(batch)
        self.assertTrue(torch.isfinite(total).item())
        self.assertTrue(torch.isfinite(items).all().item())
        total.backward()
        state = model.criterion.classification_state_dict()
        self.assertTrue(set(state) == {"pos_grad", "neg_grad"})
        model.criterion.focal_cls_loss.pos_grad.zero_()
        model.criterion.load_classification_state_dict(state)
        torch.testing.assert_close(model.criterion.focal_cls_loss.pos_grad.cpu(), state["pos_grad"])

    def test_trainer_checkpoint_restores_adaptive_statistics(self):
        model = DetectionModel("ultralytics/cfg/models/11/yolo11n.yaml", nc=32, verbose=False)
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, cls_loss="eqfl", efl_gamma=2.0, efl_scale=4.0)
        model.criterion = model.init_criterion()
        model.criterion.focal_cls_loss.pos_grad[3] = 1.25
        model.criterion.focal_cls_loss.neg_grad[3] = 2.5
        with TemporaryDirectory() as directory:
            trainer = BaseTrainer.__new__(BaseTrainer)
            trainer.model = model
            trainer.ema = SimpleNamespace(ema=deepcopy(model), updates=0)
            trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            trainer.args = SimpleNamespace(model="yolo11n.pt", close_mosaic=0)
            trainer.epoch, trainer.best_fitness, trainer.fitness = 0, 0.5, 0.5
            trainer.metrics = {}
            trainer.csv = Path(directory) / "results.csv"
            trainer.csv.write_text("epoch,metrics/mAP50(B)\n1,0.25\n", encoding="utf-8")
            trainer.last = Path(directory) / "last.pt"
            trainer.best = Path(directory) / "best.pt"
            trainer.wdir = Path(directory)
            trainer.save_period = -1
            trainer.save_model()
            checkpoint = torch.load(trainer.last, weights_only=False)
            self.assertEqual(checkpoint["train_results"]["metrics/mAP50(B)"], [0.25])
            self.assertEqual(checkpoint["cls_loss_state"]["pos_grad"][3].item(), 1.25)
            model.criterion.focal_cls_loss.pos_grad.zero_()
            model.criterion.focal_cls_loss.neg_grad.zero_()
            trainer.resume = True
            trainer.epochs = 2
            trainer.resume_training(checkpoint)
            self.assertEqual(model.criterion.focal_cls_loss.pos_grad[3].item(), 1.25)
            self.assertEqual(model.criterion.focal_cls_loss.neg_grad[3].item(), 2.5)


if __name__ == "__main__":
    unittest.main()
