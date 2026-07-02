# YOLO11 Rice Pest Next Experiment Plan

This note records the next recommended experiments after the structure-module trials.

## Current Judgment

Recent structure changes showed limited and unstable gains:

- `MEN(P3)` is still the most explainable structure-side improvement.
- `Weighted-P3Fusion`, `P2Head`, and `P3-C2PSA` did not provide stable gains across overall metrics and the three priority pests.
- Further stacking modules is not recommended before testing loss and data strategies.

The next stage should prioritize localization loss and small-object data strategy.

## Priority Metrics

Primary metrics for opening-report discussion:

- `P`
- `R`
- `mAP50`

Secondary metric:

- `mAP50-95`

Priority pest classes:

| Class ID | Class name |
|---:|---|
| 1 | Chilo suppressalis |
| 2 | Cnaphalocrocis medinalis |
| 22 | Rice plant hopper |

## Recommended Experiment Order

### 1. YOLO11 baseline + Inner-CIoU

Purpose: test whether localization loss improves small-object matching and overall recall.

Model:

```text
ultralytics/cfg/models/11/yolo11.yaml
```

Training override:

```python
iou_loss="inner_ciou"
inner_iou_ratio=0.7
```

Suggested run name:

```text
yolo11_inner-ciou_pretrained_mixed_dataset
```

### 2. YOLO11 baseline + MPDIoU

Purpose: compare another box regression loss without changing the model structure.

Model:

```text
ultralytics/cfg/models/11/yolo11.yaml
```

Training override:

```python
iou_loss="mpdiou"
```

Suggested run name:

```text
yolo11_mpdiou_pretrained_mixed_dataset
```

### 3. MEN(P3) + Inner-CIoU

Purpose: test whether the strongest structure-side candidate combines well with improved localization loss.

Model:

```text
ultralytics/cfg/models/11/yolo11-men-p3.yaml
```

Training override:

```python
iou_loss="inner_ciou"
inner_iou_ratio=0.7
```

Suggested run name:

```text
yolo11_men-p3_inner-ciou_pretrained_mixed_dataset
```

### 4. MEN(P3) + MPDIoU

Purpose: test whether MEN(P3) benefits from MPDIoU-style regression.

Model:

```text
ultralytics/cfg/models/11/yolo11-men-p3.yaml
```

Training override:

```python
iou_loss="mpdiou"
```

Suggested run name:

```text
yolo11_men-p3_mpdiou_pretrained_mixed_dataset
```

## Train Script Template

Only change `model`, `iou_loss`, and `name` between experiments.

```python
from ultralytics import YOLO

model = YOLO("/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11.yaml")
model.load("/root/ultralytics-8.3.27/yolo11n.pt")

model.train(
    data="/root/ultralytics-8.3.27/mixed_dataset/data.yaml",
    epochs=300,
    imgsz=640,
    batch=32,
    device=0,
    workers=32,
    optimizer="auto",
    patience=100,
    save=True,
    project="runs/train",
    seed=1,
    iou_loss="inner_ciou",
    inner_iou_ratio=0.7,
    name="yolo11_inner-ciou_pretrained_mixed_dataset",
)
```

Notes:

- Use `seed=1` if comparing with previous seed-1 experiments.
- Keep batch/workers consistent with the baseline whenever possible.
- `optimizer="auto"` ignores the manually specified `lr0`; do not report `lr0=0.002` as active if `optimizer="auto"` is used.

## Decision Rules

Keep an experiment only if at least one of the following is true:

1. Overall `P`, `R`, and `mAP50` improve together.
2. Overall `mAP50` improves clearly, and at least two of the three priority pests improve in `mAP50`.
3. `Rice plant hopper` improves clearly without obvious overall degradation.

Do not continue a direction if:

- overall `P` drops substantially while `R` only slightly increases;
- `mAP50-95` decreases and visual localization becomes worse;
- only one non-priority class improves while priority classes regress.

## Visualization After a Good Result

If a loss experiment improves metrics, use:

```text
find_baseline_missed_cases.py
```

to find cases where:

```text
baseline missed a small GT object
improved model detected it
```

Then use:

```text
visualize_module_effects.py
```

for feature-map and heatmap-overlay evidence.

