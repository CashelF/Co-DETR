
_base_ = './co_dino_5scale_vit_large_coco.py'

import os
# import torch
# import warnings
# # Try importing MultiScaleDeformableAttnFunction from mmcv.ops
# try:
#     from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
# except ImportError:
#     warnings.warn('Could not import MultiScaleDeformableAttnFunction to patch it for FP16.')
#     MultiScaleDeformableAttnFunction = None

# # Patch MSDA Function for FP16 support
# if MultiScaleDeformableAttnFunction is not None:
#     if not hasattr(MultiScaleDeformableAttnFunction, '_patched_for_fp16'):
#         MultiScaleDeformableAttnFunction._original_apply = MultiScaleDeformableAttnFunction.apply
        
#         # Capture class in default arg to avoid NameError after del
#         def new_apply(*args, _cls=MultiScaleDeformableAttnFunction, **kwargs):
#             # Cast all half tensors to float
#             new_args = []
#             for arg in args:
#                 if isinstance(arg, torch.Tensor) and arg.dtype == torch.float16:
#                     new_args.append(arg.float())
#                 else:
#                     new_args.append(arg)
#             return _cls._original_apply(*new_args, **kwargs)
        
#         MultiScaleDeformableAttnFunction.apply = new_apply
#         MultiScaleDeformableAttnFunction._patched_for_fp16 = True
#         print("Patched MultiScaleDeformableAttnFunction.apply for FP16 training")
    
#     # Clean up to avoid config dump error
#     del MultiScaleDeformableAttnFunction

# -------------------------
# 1) Classes / ontology
# -------------------------
# 31 Leaf nodes from maritime.json
classes = (
    'Barge', 'Bulk_Carrier', 'Container_Ship', 'Ro_Ro', 'Tanker', 
    'Cruise_Ship', 'Ferry', 'Fishing_Vessel', 'Harbor_Boat', 'Jetski', 
    'Motorboat', 'Non_Transport_Vessel', 'Paddle_Boat', 'Sailboat', 'Yacht', 
    'Buoy', 'GOPlat', 'Lighthouse', 'Obstacle', 'Unknown_Vessel', 
    'Amphib_Boat', 'Amphib_Ship', 'Carrier', 'CruDes', 'Frigate', 
    'LPV', 'Patrol', 'Submarine', 'Transport', 'Helicopter', 'Person'
)

num_classes = len(classes)

# -------------------------
# 2) Model Settings
# -------------------------
model = dict(
    backbone=dict(use_act_checkpoint=True),
    
    query_head=dict(
        num_classes=num_classes,
    ),
    
    roi_head=[dict(
        type='CoStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32, 64],
            finest_scale=56),
        bbox_head=dict(
            type='ConvFCBBoxHead',
            num_shared_convs=4,
            num_shared_fcs=1,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.05, 0.05, 0.1, 0.1]),
            reg_class_agnostic=True,
            reg_decoded_bbox=True,
            norm_cfg=dict(type='GN', num_groups=32),
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=12.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=120.0))
    )],
    
    bbox_head=[dict(
        type='CoATSSHead',
        num_classes=num_classes,
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
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=12.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=24.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=12.0)
    )],
)

# -------------------------
# 3) Data Paths
# -------------------------
data_root = '/data/cashel-data/maritime-0128'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            # --- POLICY A: Standard Resize ---
            [
                dict(
                    type='Resize',
                    img_scale=[(2000, 800), (2000, 900), (2000, 1000), (2000, 1100), (2000, 1200)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            # --- POLICY B: Large Scale Jittering (Zoom & Crop) ---
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                   # follow the original impl
                    img_scale=[(1000, 400), (3000, 1500)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(800, 800),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(2000, 800), (2000, 900), (2000, 1000), (2000, 1100), (2000, 1200)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
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
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + '/train/labels.json',
        img_prefix=data_root + '/train/',
        classes=classes,
        pipeline=train_pipeline,
        filter_empty_gt=False
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


# -------------------------
# 4) Training Params
# -------------------------
# optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)

# # Patch GIoULoss for FP16 support (Force FP32)
# try:
#     from mmdet.models.losses import GIoULoss
# except ImportError:
#     GIoULoss = None

# if GIoULoss is not None:
#     if not hasattr(GIoULoss, '_patched_for_fp16'):
#         GIoULoss._original_forward = GIoULoss.forward
        
#         def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, **kwargs):
#             if pred.dtype == torch.float16:
#                 pred = pred.float()
#             if target.dtype == torch.float16:
#                 target = target.float()
#             if weight is not None and weight.dtype == torch.float16:
#                 weight = weight.float()
#             return self._original_forward(pred, target, weight, avg_factor, reduction_override, **kwargs)
            
#         GIoULoss.forward = forward
#         GIoULoss._patched_for_fp16 = True
#         print("Patched GIoULoss for FP16 training (Force FP32)")
    
#     del GIoULoss

# fp16 = dict(loss_scale=dict(init_scale=512.))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8])

runner = dict(type='EpochBasedRunner', max_epochs=10)

# Checkpoint
load_from = 'checkpoints/co_dino_5scale_vit_large_coco.pth'

# Logging
RUN_NAME = 'codetr-large-maritime-0128-nodes'

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
                tags=['hydra-v2', 'codetr', 'large', '0128-nodes'],
            ),
            commit=True,
            with_step=True
        ),
    ],
)

# Output dir
work_dir = '/data/cashel-data/models/codetr/maritime/' + RUN_NAME

# Eval
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP', classwise=True)
checkpoint_config = dict(interval=1, save_last=True, max_keep_ckpts=2)
