# Ultralytics YOLO 🚀, AGPL-3.0 license

from types import SimpleNamespace

import torch

from ultralytics.nn.modules import DAGGDetect
from ultralytics.nn.tasks import DetectionModel


def build_dagg_model():
    """Build a small DAGG model with the loss hyperparameters required by the detection loss."""
    model = DetectionModel("ultralytics/cfg/models/11/yolo11-dagg-p3.yaml", verbose=False)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5, iou_loss="ciou", inner_iou_ratio=0.7)
    return model


def test_dagg_training_forward_and_loss():
    """DAGG should add a finite fourth loss item while preserving normal detection outputs."""
    model = build_dagg_model().train()
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0, 0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0], [22.0]]),
        "bboxes": torch.tensor(
            [
                [0.30, 0.30, 0.08, 0.06],
                [0.36, 0.34, 0.05, 0.05],
                [0.70, 0.65, 0.12, 0.10],
            ]
        ),
    }

    predictions = model(batch["img"])
    assert isinstance(predictions, tuple)
    assert len(predictions[0]) == 3
    assert predictions[1].shape == (2, 1, 8, 8)

    loss, loss_items = model(batch, preds=predictions)
    assert loss_items.shape == (4,)
    assert torch.isfinite(loss)
    assert torch.isfinite(loss_items).all()
    assert loss_items[-1] > 0


def test_dagg_is_skipped_during_inference():
    """Evaluation should use the standard Detect path without executing the auxiliary predictor."""
    model = build_dagg_model().eval()
    head = model.model[-1]
    assert isinstance(head, DAGGDetect)
    calls = []
    hook = head.dagg_head.register_forward_hook(lambda *args: calls.append(True))
    try:
        with torch.no_grad():
            predictions = model(torch.rand(1, 3, 64, 64))
    finally:
        hook.remove()

    assert isinstance(predictions, tuple)
    assert predictions[0].shape[0] == 1
    assert calls == []
