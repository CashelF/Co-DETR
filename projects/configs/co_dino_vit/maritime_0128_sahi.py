_base_ = './maritime_0128_nodes.py'

# 0. Classes (Copied from base to avoid import issues)
classes = (
    'Barge', 'Bulk_Carrier', 'Container_Ship', 'Ro_Ro', 'Tanker', 
    'Cruise_Ship', 'Ferry', 'Fishing_Vessel', 'Harbor_Boat', 'Jetski', 
    'Motorboat', 'Non_Transport_Vessel', 'Paddle_Boat', 'Sailboat', 'Yacht', 
    'Buoy', 'GOPlat', 'Lighthouse', 'Obstacle', 'Unknown_Vessel', 
    'Amphib_Boat', 'Amphib_Ship', 'Carrier', 'CruDes', 'Frigate', 
    'LPV', 'Patrol', 'Submarine', 'Transport', 'Helicopter', 'Person'
)

# 1. Pipeline Definition
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Pre-pipeline (Load Image + Anno)
train_pipeline_pre = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True)
]

# Mode A Pipeline (Global Context) -> Resize to 1024x1024
train_pipeline_a = [
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# Mode B Pipeline (Micro-Zoom Slicing) 
# Input to this is ALREADY CROPPED by HybridMosaicDataset
# So just Resize UP to 1024x1024
train_pipeline_b = [
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# Test Pipeline (Keep Resolution High for Slicing)
# We set scale to original image size (1920x1080) to preserve details
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080), # Target scale (should match original mostly)
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

# 2. Dataset Definition
data_root = '/data/cashel-data/maritime-0128'

data = dict(
    _delete_=True, # Force replacement of data dict to avoid inheritance issues
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='HybridMosaicDataset',
        # Inner dataset
        dataset=dict(
            type='CocoDataset',
            ann_file=data_root + '/train/labels.json',
            img_prefix=data_root + '/train/',
            classes=classes,
            pipeline=train_pipeline_pre,
            filter_empty_gt=False
        ),
        pipeline_a=train_pipeline_a,
        pipeline_b=train_pipeline_b,
        prob=0.5,
        crop_size=(480, 480), # The size of the crop taken from original image
        img_scale=(1024, 1024) # The size input to the model
    ),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + '/val/labels.json',
        img_prefix=data_root + '/val/',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + '/test/labels.json',
        img_prefix=data_root + '/test/',
        classes=classes,
        pipeline=test_pipeline
    ),
)

# 3. Model Definition
model = dict(
    type='SahiCoDETR',
    # SAHI Configuration
    sahi_cfg=dict(
        crop_size=(480, 480),
        stride=320,
        target_size=(1024, 1024)
    )
)

RUN_NAME = 'codetr-large-maritime-0128-sahi'

WANDB_PROJECT = 'co-detr-hydra'
WANDB_ENTITY  = 'cashel'
WANDB_RUNNAME = RUN_NAME

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project=WANDB_PROJECT,
                name=WANDB_RUNNAME,
                entity=WANDB_ENTITY,
                tags=['hydra-v2', 'codetr', 'large', '0128-sahi'],
            ),
            commit=True,
            with_step=True
        ),
    ],
)
work_dir = '/data/cashel-data/models/codetr/maritime/' + RUN_NAME
