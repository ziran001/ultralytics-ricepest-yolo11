from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(
    data='/root/ultralytics-8.3.27/mixed_dataset/data.yaml',
    epochs=300,
    imgsz=640,
    batch=32,
    device=0,
    workers=32,
    lr0=0.002,
    optimizer='auto',
    patience=100,
    save=True,
    project='runs/train',
    name='yolo11_mixed_dataset'
)


# from ultralytics import YOLO

# model = YOLO('yolo11n.pt')

# model.train(
#     data='/root/ultralytics-8.3.27/mixed_dataset_1+1_sam_aug_goodrgba_v2_200/data.yaml',
#     epochs=400,                    # 从 300 增加到 400
#     imgsz=640,
#     batch=32,
#     device=0,
#     workers=32,
#     lr0=0.002,
#     optimizer='auto',
#     patience=100,
#     save=True,
#     project='runs/train',
#     name='yolo11_1+1_aug',         # 改个名字区分
    
#     # ========== 新增：强数据增强参数 ==========
#     # 颜色增强（模拟不同光照）
#     hsv_h=0.1,        # 色调增强（默认0.015）
#     hsv_s=0.8,        # 饱和度增强（默认0.7）
#     hsv_v=0.5,        # 明度增强（默认0.4）
    
#     # 几何增强（增加姿态多样性）
#     degrees=30,       # 随机旋转 ±30°（默认0）
#     translate=0.2,    # 随机平移 20%（默认0.1）
#     scale=0.7,        # 随机缩放 0.7~1.3（默认0.5）
#     shear=10,         # 随机剪切 ±10°（默认0）
#     perspective=0.0005,  # 随机透视（默认0）
#     flipud=0.2,       # 上下翻转概率（默认0）
#     fliplr=0.5,       # 左右翻转概率（默认0.5）
    
#     # 混合增强（关键：帮助飞虱泛化）
#     mosaic=1.0,       # mosaic 概率（默认1.0）
#     mixup=0.3,        # mixup 概率（默认0）- 让飞虱与其他背景混合
#     copy_paste=0.3,   # copy-paste 概率（默认0）- 复制飞虱到不同位置
    
#     # 正则化
#     erasing=0.1,      # 随机擦除概率（新增，帮助飞虱关注局部特征）
# )