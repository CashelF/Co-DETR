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
data_root = '/data/cashel-data/gladius-classification-expanded/'
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
    samples_per_gpu=1,       # ViT-L is chunky. Start at 1; increase if you truly have VRAM.
    workers_per_gpu=4,
    train=dict(
        type='CocoDataset',
        ann_file=data_root + train_anno,
        img_prefix=data_root + 'train/' + img_dir,
        classes=classes,
        filter_empty_gt=False,    # keep images with no boxes
        # if your base defines a custom train_pipeline, you can pull it in via {{_base_.train_pipeline}}
        # otherwise, leave it to the base file.
        # pipeline={{_base_.train_pipeline}},
    ),
    val=dict(
        type='CocoDataset',
        ann_file=data_root + val_anno,
        img_prefix=data_root + 'val/' + img_dir,
        classes=classes,
        test_mode=True,
        # pipeline={{_base_.test_pipeline}},
    ),
    test=dict(
        type='CocoDataset',
        ann_file=data_root + val_anno,
        img_prefix=data_root + 'val/' + img_dir,
        classes=classes,
        test_mode=True,
        # pipeline={{_base_.test_pipeline}},
    ),
)

# -------------------------
# 5) Pretrained checkpoint (COCO objects, ViT-L Co-DINO/Co-DETR)
#    Put the .pth under ./checkpoints and point here.
# -------------------------
load_from = 'checkpoints/co_dino_5scale_vit_large_coco.pth'

# -------------------------
# 6) Optim/schedule:
#    Keep your ViT-L base’s optimizer & layer decay unless you want to override.
#    If your *effective* total batch < base, scale lr proportionally.
# -------------------------
# Example override if you need it (commented; base likely already sets this):
# optimizer = dict(
#     type='AdamW',
#     lr=5e-5,
#     weight_decay=0.01,
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.8),
# )
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.01, step=[7])
# runner = dict(type='EpochBasedRunner', max_epochs=12)
# --- Mixed precision (AMP) ---
# fp16 = dict(loss_scale='dynamic')
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
custom_imports = dict(
    imports=[
#         'projects.hooks.patch_msda_hook',
#         'projects.hooks.force_pt_msda_hook',
#         'projects.hooks.freeze_backbone',
        #   'projects.optim.oss_adamw',
    ],
    allow_failed_imports=False,
)

custom_hooks = [
#     dict(type='PatchMSDAHook'),
#     dict(type='ForcePTMSDAHook'),
    #   dict(type='FreezeBackboneHook', module_names=['backbone']),
]

# optimizer = dict(
#     type='OSSAdamW',   
#     lr=5e-5,
#     weight_decay=0.01,
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.8),
# )



# -------------------------
# 7) Logging (MMDet v2) — same W&B hook pattern you used
# -------------------------
WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'co-detr-hydra')
WANDB_ENTITY  = os.getenv('WANDB_ENTITY',  'cashel')
WANDB_RUNNAME = os.getenv('WANDB_RUN_NAME', 'gladius_vitl_abe_data_prev_decoder')

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
                notes='Co-DINO/Co-DETR ViT-L on custom 9-class dataset (MMDet v2).'
            ),
            commit=True,
            with_step=True
        ),
    ],
)

# -------------------------
# 8) Eval / checkpoints / work_dir (same QoL as R50)
# -------------------------
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP', classwise=True)
checkpoint_config = dict(interval=1, save_last=True, max_keep_ckpts=2)
work_dir = './work_dirs/gladius_vitl'
