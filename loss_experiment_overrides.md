# Loss Experiment Overrides

Copy the matching lines into `train_11.py`.

## YOLO11 baseline + DAGG(P3, training-only)

```python
model = YOLO("/root/ultralytics-8.3.27/ultralytics/cfg/models/11/yolo11-dagg-p3.yaml")
name="yolo11_dagg-p3_pretrained_mixed_dataset",
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

