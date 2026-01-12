# File: projects/configs/codino/gladius_vitl_12e.py
_base_ = './co_dino_5scale_vit_large_coco.py'

import os

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
data_root = '/data/cashel-data/abes-gladius-data-vid/'
train_anno = 'train/labels.json'
val_anno   = 'val/labels.json'
img_dir    = ''

# -------------------------
# 3) Make it 9 classes everywhere (list items must be fully replaced)
#    Keep the same head types as in your ViT-L base.
# -------------------------
model = dict(
    backbone=dict(use_act_checkpoint=True),

    # CoDINO/CoDETR DETR-like head
    query_head=dict(
        # num_query=600,
        # dn_cfg=dict(
        #     group_cfg=dict(num_dn_queries=200)
        # ),
        # transformer=dict(
        #     num_feature_levels=5,
        #     decoder=dict(transformerlayers=dict(
        #         attn_cfgs=[
        #             dict(type='MultiheadAttention', embed_dims=256, num_heads=8, dropout=0.0), 
        #             dict(type='MultiScaleDeformableAttention', embed_dims=256, num_levels=5)
        #         ]
        #     ))
        # ),
        num_classes=len(classes),
        num_track_queries=300,  # [NEW] Persistent track slots
        spawn_score_thresh=0.4, # [NEW] Threshold to spawn new track
        miss_tolerance=5,       # [NEW] Frames to keep lost track
        track_loss_weight=1.0,  # [NEW] Weight for track supervision
        # query_init_checkpoint='codetr-chkpt.pt', # [NEW] Init queries from checkpoint
        query_init_checkpoint=None,
        transformer=dict(
            decoder=dict(transformerlayers=dict(
                type='TrackingDetrTransformerDecoderLayer',
                attn_cfgs=[
                    dict(type='TrackingMultiheadAttention', embed_dims=256, num_heads=8, dropout=0.0), 
                    dict(type='MultiScaleDeformableAttention', embed_dims=256, num_levels=5, dropout=0.0)
                ]
            ))
        ),
    ),


    # ROI head list (base has one element; we replace it fully to set num_classes)
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            # ViT-L base uses strides [4,8,16,32,64]
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56),
        bbox_head=dict(
            # ViT-L base uses ConvFCBBoxHead for RoI
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=len(classes),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='GN', num_groups=32),
            # keep loss magnitudes from your base (num_dec_layer=6, lambda_2=2.0)
            # → 1.0 * 6 * 2.0 = 12.0 ; 10.0 * 6 * 2.0 = 120.0
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=12.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=120.0))
    )],

    # Co-ATSS auxiliary head (list with one element; replace fully to set num_classes)
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
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        # keep focal/GIoU/centerness weights consistent with the ViT-L base:
        # 1.0 * 6 * 2.0 = 12.0 ; 2.0 * 6 * 2.0 = 24.0
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=12.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0)
    )],
)

# -------------------------
# 4) Datasets — same behavior as R50:
#    - declare classes
#    - keep negatives (filter_empty_gt=False) for train
#    - point prefixes correctly
# -------------------------
data = dict(
    samples_per_gpu=2,       # ViT-L is chunky. Start at 1; increase if you truly have VRAM.
    workers_per_gpu=4,
    train=dict(
        type='CocoVideoDataset',
        ann_file=data_root + train_anno,
        img_prefix=data_root + 'train/' + img_dir,
        classes=classes,
        load_as_video=True,
        filter_empty_gt=False,    # keep images with no boxes
        # if your base defines a custom train_pipeline, you can pull it in via {{_base_.train_pipeline}}
        # otherwise, leave it to the base file.
        # pipeline={{_base_.train_pipeline}},
    ),
    val=dict(
        type='CocoVideoDataset',
        ann_file=data_root + val_anno,
        img_prefix=data_root + 'val/' + img_dir,
        classes=classes,
        load_as_video=True,
        test_mode=True,
        # pipeline={{_base_.test_pipeline}},
    ),
    test=dict(
        type='CocoVideoDataset',
        ann_file=data_root + val_anno,
        img_prefix=data_root + 'val/' + img_dir,
        classes=classes,
        load_as_video=True,
        test_mode=True,
        # pipeline={{_base_.test_pipeline}},
    ),
)

# -------------------------
# 5) Pretrained checkpoint (COCO objects, ViT-L Co-DINO/Co-DETR)
#    Put the .pth under ./checkpoints and point here.
# -------------------------
load_from = 'codetr-chkpt.pt'

# -------------------------
# 6) Optim/schedule:
#    Keep your ViT-L base’s optimizer & layer decay unless you want to override.
#    If your *effective* total batch < base, scale lr proportionally.
# -------------------------
# Example override if you need it (commented; base likely already sets this):
optimizer = dict(
    type='AdamW',
    lr=1e-5,
    weight_decay=0.01,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.8),
)
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=1e-6
)

runner = dict(type='EpochBasedRunner', max_epochs=24)
# --- Mixed precision (AMP) ---
# fp16 = dict(loss_scale='dynamic')
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))


# -------------------------
# 8) Eval / checkpoints / work_dir (same QoL as R50)
# -------------------------
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP', classwise=True)
checkpoint_config = dict(interval=1, save_last=True, max_keep_ckpts=2)
work_dir = './work_dirs/gladius_vitl'

# -------------------------
# 7) Logging (MMDet v2) — same W&B hook pattern you used
# -------------------------
WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'co-detr-hydra')
WANDB_ENTITY  = os.getenv('WANDB_ENTITY',  'cashel')
WANDB_RUNNAME = os.getenv('WANDB_RUN_NAME', 'gladius_vitl_abes_track_queries')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# Define pipelines
val_loss_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSeqInfo'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(2048, 1280), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'video_id', 'frame_id', 'is_video_first')
    )
]


# -------------------------
# Test Pipeline (copy structure from base or define here)
# -------------------------
train_eval_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSeqInfo'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1280),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                           'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg', 'video_id', 'frame_id', 'is_video_first')
            )
        ])
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSeqInfo'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 2400), (512, 2400), (544, 2400), (576, 2400),
                               (608, 2400), (640, 2400), (672, 2400), (704, 2400),
                               (736, 2400), (768, 2400), (800, 2400), (832, 2400),
                               ],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 2400), (512, 2400), (544, 2400), (576, 2400),
                               (608, 2400), (640, 2400), (672, 2400), (704, 2400),
                               (736, 2400), (768, 2400), (800, 2400), (832, 2400),
                               ],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg', 'video_id', 'frame_id', 'is_video_first')
    )
]

# Update the main data pipeline to use this new one
data['train']['pipeline'] = train_pipeline

# Datasets
val_loss_dataset = dict(
    type='CocoVideoDataset',
    ann_file=data_root + val_anno,
    img_prefix=data_root + 'val/' + img_dir,
    classes=classes,
    load_as_video=True,
    test_mode=False,
    filter_empty_gt=False,
    pipeline=val_loss_pipeline
)

train_eval_dataset = dict(
    type='CocoVideoDataset',
    ann_file=data_root + train_anno,
    img_prefix=data_root + 'train/' + img_dir,
    classes=classes,
    load_as_video=True,
    test_mode=True,
    filter_empty_gt=False,
    pipeline=train_eval_pipeline
)


custom_imports = dict(
    imports=[
          'mmdet.datasets.coco_video',
          'projects.core.val_loss_hook',
          'projects.core.train_eval_hook',
          'projects.core.pipeline'
    ],
    allow_failed_imports=False,
)

custom_hooks = [
    dict(
        type='ExpMomentumEMAHook',
        momentum=0.0001,
        priority=49),
    dict(
        type='ValLossHook',
        val_dataset_cfg=val_loss_dataset,
        interval=1,
        priority='NORMAL'
    ),
    dict(
        type='TrainEvalHook',
        dataset_cfg=train_eval_dataset,
        interval=1,
        subset_ratio=0.20,
        priority='NORMAL',
        metric='bbox', 
        classwise=True
    )
]

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
                tags=['co-detr', 'codino', 'vit-large', 'gladius', '9-classes'],
                notes='Co-DINO/Co-DETR ViT-L with Val Loss + Train mAP (20%).'
            ),
            commit=True,
            with_step=True
        ),
    ],
)


