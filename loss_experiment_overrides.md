# Loss Experiment Overrides

Copy the matching lines into `train_11.py`.

## YOLO11 baseline + DAGG(P3, training-only)

```python
model = YOLO("/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11-dagg-p3.yaml")
name="yolo11_dagg-p3_pretrained_mixed_dataset",
```

YAML head arguments:

```yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.25, 0.25, 1.0, 0.01, 4.0, 0.5, 0.1, 0.25]]
```

Argument order:

```text
[loss_gain, sigma_scale, min_sigma, size_reference, max_size_weight, density_gain, density_radius, background_gain]
```

## DAGG(P3) loss gain ablation

Use the same training script as DAGG(P3), but change the model path to the matching YAML for each `loss_gain`.

```yaml
# yolo11-dagg-p3-gain010.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.10, 0.25, 1.0, 0.01, 4.0, 0.5, 0.1, 0.25]]

# yolo11-dagg-p3-gain025.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.25, 0.25, 1.0, 0.01, 4.0, 0.5, 0.1, 0.25]]

# yolo11-dagg-p3-gain050.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.50, 0.25, 1.0, 0.01, 4.0, 0.5, 0.1, 0.25]]
```

Suggested names:

```python
name="yolo11_dagg-p3_gain010_pretrained_mixed_dataset",
name="yolo11_dagg-p3_gain025_pretrained_mixed_dataset",
name="yolo11_dagg-p3_gain050_pretrained_mixed_dataset",
```

## DAGG(P3) sigma scale ablation

Run this after selecting the best `loss_gain`. Change the model path to the matching YAML.

Current next run after `gain010`:

```python
model = YOLO("/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11-dagg-p3-gain010-sigma035.yaml")
name="yolo11_dagg-p3_gain010_sigma035_pretrained_mixed_dataset",
```

```yaml
# yolo11-dagg-p3-sigma015.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.25, 0.15, 1.0, 0.01, 4.0, 0.5, 0.1, 0.25]]

# yolo11-dagg-p3-sigma025.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.25, 0.25, 1.0, 0.01, 4.0, 0.5, 0.1, 0.25]]

# yolo11-dagg-p3-sigma035.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.25, 0.35, 1.0, 0.01, 4.0, 0.5, 0.1, 0.25]]

# yolo11-dagg-p3-gain010-sigma035.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.10, 0.35, 1.0, 0.01, 4.0, 0.5, 0.1, 0.25]]
```

Suggested names:

```python
name="yolo11_dagg-p3_sigma015_pretrained_mixed_dataset",
name="yolo11_dagg-p3_sigma025_pretrained_mixed_dataset",
name="yolo11_dagg-p3_sigma035_pretrained_mixed_dataset",
```

## DAGG(P3) density ablation

```yaml
# yolo11-dagg-p3-density0.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.25, 0.25, 1.0, 0.01, 4.0, 0.0, 0.1, 0.25]]

# yolo11-dagg-p3-radius005.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.25, 0.25, 1.0, 0.01, 4.0, 0.5, 0.05, 0.25]]

# yolo11-dagg-p3-radius015.yaml
- [[16, 19, 22], 1, DAGGDetect, [nc, 0.25, 0.25, 1.0, 0.01, 4.0, 0.5, 0.15, 0.25]]
```

Suggested names:

```python
name="yolo11_dagg-p3_density0_pretrained_mixed_dataset",
name="yolo11_dagg-p3_radius005_pretrained_mixed_dataset",
name="yolo11_dagg-p3_radius015_pretrained_mixed_dataset",
```

## YOLO11 baseline + Inner-CIoU

```python
model = YOLO("/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11.yaml")
iou_loss="inner_ciou",
inner_iou_ratio=0.7,
name="yolo11_inner-ciou_pretrained_mixed_dataset",
```

## YOLO11 baseline + MPDIoU

```python
model = YOLO("/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11.yaml")
iou_loss="mpdiou",
name="yolo11_mpdiou_pretrained_mixed_dataset",
```

## MEN(P3) + Inner-CIoU

```python
model = YOLO("/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11-men-p3.yaml")
iou_loss="inner_ciou",
inner_iou_ratio=0.7,
name="yolo11_men-p3_inner-ciou_pretrained_mixed_dataset",
```

## MEN(P3) + MPDIoU

```python
model = YOLO("/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11-men-p3.yaml")
iou_loss="mpdiou",
name="yolo11_men-p3_mpdiou_pretrained_mixed_dataset",
```

