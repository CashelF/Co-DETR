# File: projects/configs/co_deformable_detr/gladius_r50_1x.py
import os
_base_ = './co_deformable_detr_r50_1x_coco.py'

# -------------------------
# 1) Classes / ontology
# -------------------------
classes = (
    'Person',
    'Vehicle',
    'Vehicle/Civilian/Motorcycle',
    'Vehicle/Civilian/Pickup',
    'Vehicle/Civilian/SUV',
    'Vehicle/Civilian/Sedan',
    'Vehicle/Civilian/Truck',
    'Vehicle/Civilian/Unknown',
    'Vehicle/Civilian/Van',
)

# -------------------------
# 2) Paths (COCO-style)
# -------------------------
data_root = '/data/cashel-data/gladius-classification-expanded/'
train_anno = 'train/labels.json'
val_anno   = 'val/labels.json'
img_dir    = ''

# -------------------------
# 3) Model: set num_classes
# Base config defines the head; we only override class count.
# -------------------------
model = dict(
    # 1) DETR head (dict → deep-merge works)
    query_head=dict(num_classes=len(classes)),

    # 2) ROI head (LIST → must replace element with a full, typed dict)
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64],
            finest_scale=112),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=len(classes),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            reg_decoded_bbox=True,
            # keep losses consistent with base (num_dec_layer=6, lambda_2=2.0 → 120.0)
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=120.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=120.0))
    )],

    # 3) Co-ATSS aux head (LIST → same story)
    bbox_head=[dict(
        type='CoATSSHead',
        num_classes=len(classes),
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        # keep losses consistent with base (12.0 / 24.0 / 12.0)
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=12.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0)
    )],
)

# -------------------------
# 4) Datasets
# -------------------------
data = dict(
    samples_per_gpu=2,     # per-GPU; tune for your VRAM
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + train_anno,
        img_prefix=data_root + "train/" + img_dir,
        classes=classes,
        filter_empty_gt=False,   # keep images with no boxes
    ),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + val_anno,
        img_prefix=data_root + "val/" + img_dir,
        classes=classes,
        test_mode=True,
    ),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + val_anno,
        img_prefix=data_root + "val/" + img_dir,
        classes=classes,
        test_mode=True,
    ),
)

# -------------------------
# 5) Pretrained checkpoint (strongly recommended)
# Drop the downloaded Co-Deformable-DETR R50 COCO checkpoint into:
#   Co-DETR/checkpoints/co_deformable_detr_r50_1x_coco.pth
# -------------------------
load_from = 'checkpoints/co_deformable_detr_r50_1x_coco.pth'  # path inside the repo

# -------------------------
# 6) Schedule & logging tweaks (optional)
# The base file is 1x (12 epochs) with AdamW. If your total batch < 16,
# scale LR linearly (e.g., total batch 4 => lr * 0.25). Here I leave base LR
# alone—change if you reduce total batch a lot.
# -------------------------
# optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.05)  # uncomment to override
# lr_config = dict(policy='step', step=[8, 11])                  # from base
# runner = dict(type='EpochBasedRunner', max_epochs=12)          # from base

# More frequent logs are handy when debugging custom data:
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])

# Save best by bbox mAP each epoch:
evaluation = dict(interval=1, metric='bbox', classwise=True)
checkpoint_config = dict(interval=1, save_last=True, max_keep_ckpts=2)

WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'co-detr-hydra')
WANDB_ENTITY  = os.getenv('WANDB_ENTITY',  'cashel')
WANDB_RUNNAME = os.getenv('WANDB_RUN_NAME', 'gladius_r50')

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
                tags=['co-detr', 'gladius', 'r50'],
                notes='Co-DETR (MMDet v2)'
            ),
            commit=True,          # log every interval
            with_step=True
        ),
    ],
)

# Save best checkpoint by bbox mAP every epoch (MMDet v2)
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP')  # keep classwise in eval if you like
checkpoint_config = dict(interval=1, save_last=True, max_keep_ckpts=2)

# Work dir (so you don’t stomp the COCO run)
work_dir = './work_dirs/gladius_r50_1x'


# tools/dist_train.sh projects/configs/co_deformable_detr/gladius_r50_1x.py 8 work_dir/gladius