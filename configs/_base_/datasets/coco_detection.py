dataset_type = 'CocoDataset'
data_root = '/work/u5216579/ctr/data/PCB_v3/'#'/home/u5216579/vf/data/coco/' #'data/coco/'
img_norm_cfg = dict(
    #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[38.720, 51.155, 40.22], std=[53.275, 52.273, 46.819], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='Sharpness', prob=0.0, level=8),
    dict(type='Rotate', prob=0.75, level=10, max_rotate_angle=360),
    dict(type='Color', prob=0.6, level=6),
    dict(type='ColorTransform', level=4.0, prob=0.5),
    dict(type='BrightnessTransform', level=4.0, prob=0.5),
    dict(type='ContrastTransform', level=4.0, prob=0.5),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    
    #train=dict(
    #    type=dataset_type,
    #    ann_file=data_root + 'annotations/instances_train2017.json',
    #    img_prefix=data_root + 'train2017/',
    #    pipeline=train_pipeline),
    #val=dict(
    #    type=dataset_type,
    #    ann_file=data_root + 'annotations/instances_val2017.json',
    #    img_prefix=data_root + 'val2017/',
    #    pipeline=test_pipeline),
    #test=dict(
    #    type=dataset_type,
    #    ann_file=data_root + 'annotations/instances_val2017.json',
    #    img_prefix=data_root + 'val2017/',
    #    pipeline=test_pipeline))
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        filter_empty_gt=False),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
