# YOLO11 Rice Pest Next Experiment Plan

This note records the next recommended experiments after the structure-module trials.

## Current Judgment

Recent structure changes showed limited and unstable gains:

- `MEN(P3)` is still the most explainable structure-side improvement.
- `Weighted-P3Fusion`, `P2Head`, and `P3-C2PSA` did not provide stable gains across overall metrics and the three priority pests.
- `DAGG(P3, training-only)` is a positive training-supervision result, but it still needs parameter ablation before it should be stacked with other modules.
- Further stacking inference-time modules is not recommended before testing loss, training-supervision, and data strategies.

The next stage should prioritize DAGG ablation, localization loss, and small-object data strategy.

## Latest DAGG Result

Model:

```text
ultralytics/cfg/models/11/yolo11-dagg-p3.yaml
```

Run:

```text
runs/train/yolo11_dagg-p3_pretrained_mixed_dataset
```

Summary:

```text
YOLO11-dagg-p3 summary (fused): 242 layers, 2,595,886 parameters, 0 gradients, 6.3 GFLOPs
```

| Class | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| all | 0.737 | 0.691 | 0.732 | 0.485 |
| Chilo suppressalis | 0.846 | 0.861 | 0.905 | 0.609 |
| Cnaphalocrocis medinalis | 0.796 | 0.793 | 0.808 | 0.567 |
| Rice plant hopper | 0.870 | 0.900 | 0.928 | 0.733 |

Interpretation:

- Compared with the YOLO11 baseline, DAGG improves all overall metrics.
- The strongest class-level signal is `Rice plant hopper`, where `mAP50-95` reaches `0.733`.
- `Chilo suppressalis` recall improves, but `mAP50-95` is still slightly below the baseline, so localization quality remains the key risk.
- Because this is a single run, treat it as a promising direction rather than a final model choice.

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

### 1. DAGG loss gain ablation

Purpose: check whether the current Gaussian guidance is too weak or too strong.

Base model:

```text
ultralytics/cfg/models/11/yolo11-dagg-p3.yaml
```

Change the first `DAGGDetect` argument after `nc`:

```text
[loss_gain, sigma_scale, min_sigma, size_reference, max_size_weight, density_gain, density_radius, background_gain]
```

Run these values:

```text
loss_gain = 0.10, 0.25, 0.50
```

Model files:

```text
ultralytics/cfg/models/11/yolo11-dagg-p3-gain010.yaml
ultralytics/cfg/models/11/yolo11-dagg-p3.yaml
ultralytics/cfg/models/11/yolo11-dagg-p3-gain050.yaml
```

Suggested run names:

```text
yolo11_dagg-p3_gain010_pretrained_mixed_dataset
yolo11_dagg-p3_gain025_pretrained_mixed_dataset
yolo11_dagg-p3_gain050_pretrained_mixed_dataset
```

Primary decision signal:

- keep the setting that improves `all mAP50-95` while recovering `Chilo suppressalis mAP50-95`;
- reduce `loss_gain` if precision drops or the priority pest localization metrics regress.

### 2. DAGG sigma scale ablation

Purpose: check whether the Gaussian target should be sharper or smoother.

After choosing the best `loss_gain`, run:

```text
sigma_scale = 0.15, 0.25, 0.35
```

Model files:

```text
ultralytics/cfg/models/11/yolo11-dagg-p3-sigma015.yaml
ultralytics/cfg/models/11/yolo11-dagg-p3.yaml
ultralytics/cfg/models/11/yolo11-dagg-p3-sigma035.yaml
```

Suggested run names:

```text
yolo11_dagg-p3_sigma015_pretrained_mixed_dataset
yolo11_dagg-p3_sigma025_pretrained_mixed_dataset
yolo11_dagg-p3_sigma035_pretrained_mixed_dataset
```

Primary decision signal:

- smaller `sigma_scale` should help center precision if it does not hurt recall;
- larger `sigma_scale` is preferred if the current setting over-concentrates supervision and hurts small-object matching.

### 3. DAGG density ablation

Purpose: verify whether density-aware weighting helps dense rice-pest scenes or adds background noise.

Run:

```text
density_gain = 0.0
density_radius = 0.05, 0.10, 0.15
```

Model files:

```text
ultralytics/cfg/models/11/yolo11-dagg-p3-density0.yaml
ultralytics/cfg/models/11/yolo11-dagg-p3-radius005.yaml
ultralytics/cfg/models/11/yolo11-dagg-p3.yaml
ultralytics/cfg/models/11/yolo11-dagg-p3-radius015.yaml
```

Suggested run names:

```text
yolo11_dagg-p3_density0_pretrained_mixed_dataset
yolo11_dagg-p3_radius005_pretrained_mixed_dataset
yolo11_dagg-p3_radius010_pretrained_mixed_dataset
yolo11_dagg-p3_radius015_pretrained_mixed_dataset
```

Primary decision signal:

- keep density weighting only if it improves `Rice plant hopper` and does not reduce `Chilo suppressalis mAP50-95`;
- disable or weaken density weighting if dense-background images create false positives.

### 4. MEN(P3) + DAGG

Purpose: test whether P3 texture enhancement and training-only Gaussian guidance are complementary.

Model target:

```text
ultralytics/cfg/models/11/yolo11-men-p3-dagg.yaml
```

Suggested run name:

```text
yolo11_men-p3_dagg_pretrained_mixed_dataset
```

Continue only if:

- `all mAP50-95 >= 0.489`; or
- `Chilo suppressalis mAP50-95 >= 0.612` without overall degradation.

### 5. MEN(P3) + DySample + DAGG

Purpose: reserve this as the strongest combination candidate after standalone DAGG and `MEN(P3)+DAGG` are stable.

Model target:

```text
ultralytics/cfg/models/11/yolo11-men-p3-dysample-dagg.yaml
```

Suggested run name:

```text
yolo11_men-p3_dysample_dagg_pretrained_mixed_dataset
```

Run this only if at least one of the earlier DAGG combination experiments is positive.

### 6. YOLO11 baseline + Inner-CIoU

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

### 7. YOLO11 baseline + MPDIoU

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

### 8. MEN(P3) + Inner-CIoU

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

### 9. MEN(P3) + MPDIoU

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
- For DAGG experiments, change the `DAGGDetect` arguments in the YAML and keep the rest of the training script fixed.

## Decision Rules

Keep an experiment only if at least one of the following is true:

1. Overall `P`, `R`, and `mAP50` improve together.
2. Overall `mAP50` improves clearly, and at least two of the three priority pests improve in `mAP50`.
3. `Rice plant hopper` improves clearly without obvious overall degradation.
4. For DAGG experiments, `all mAP50-95` improves and `Chilo suppressalis mAP50-95` does not fall further below the baseline.

Do not continue a direction if:

- overall `P` drops substantially while `R` only slightly increases;
- `mAP50-95` decreases and visual localization becomes worse;
- only one non-priority class improves while priority classes regress.
- DAGG improves `Rice plant hopper` only by sacrificing `Chilo suppressalis` localization.

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

