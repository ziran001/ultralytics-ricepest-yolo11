import torch

from ultralytics.cfg import get_cfg
from ultralytics.utils.metrics import bbox_inner_iou, bbox_iou


cfg = get_cfg(overrides={"iou_loss": "inner_ciou", "inner_iou_ratio": 0.7})
assert cfg.iou_loss == "inner_ciou"
assert cfg.inner_iou_ratio == 0.7

pred = torch.tensor([[10.0, 12.0, 30.0, 36.0]], requires_grad=True)
target = torch.tensor([[12.0, 10.0, 31.0, 35.0]])

ciou = bbox_iou(pred, target, xywh=False, CIoU=True)
inner_ciou_at_one = bbox_inner_iou(pred, target, ratio=1.0, xywh=False, CIoU=True)
torch.testing.assert_close(inner_ciou_at_one, ciou, rtol=1e-5, atol=1e-6)

loss = 1.0 - bbox_inner_iou(pred, target, ratio=0.7, xywh=False, CIoU=True)
loss.sum().backward()
assert torch.isfinite(loss).all()
assert pred.grad is not None and torch.isfinite(pred.grad).all()

print("Inner-CIoU checks passed.")
