
_base_ = './co_dino_5scale_vit_large_coco.py'

import os

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
train_dirs = [
    '/data/abe-data/cutlass-node-annotations-node-only/Aerial Drone View of US Navy Patrol Boat YP707 - Delaware River - Philadelphia_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Aerial Sailboat and Seascapes Collection_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Aerial View of Pieter and Steve Sailing on a Hobie 21SE_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Aerial View of USS Stockdale Guided Missile Destroyer and USS Princeton Guided Missile Cruiser_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Chinese Navy Hospital Ship Peace Ark 866_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/DJI Phantom 3 - Cargo ship Arborella at Sea - Aerial Footage_1fps.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/DOD_105991467_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/HMAS Canberra (L02) - Forty Ships and Submarines Steam in Close Formation During RIMPAC_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Jet ski Chase Drone Aerial Video_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/LPV_Abandoned_1fps.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/LPV_Caribbean_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/LPV_Stratton_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/LPV_Tampa_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/NATO Allies conduct drills to hunt submarines in exercise Dynamic Manta_1fps (classified)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/NATO Allies practise submarine hunting in Exercise Dynamic Manta_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Oil Platform Dive Trip DRONE_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Oil Rig Esther, Seal Beach, Aerial_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_135416449.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_135550391.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_135649557.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_135837471.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_135900107.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_135921580.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_135944865.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_140009820.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_140021344.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_140101108.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_141735049.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_141926666.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_142245804.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_142325694-2.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_142633972.TS_1fps.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_142817424.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_143739916-2.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_144117978.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_144155493.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_145301960-2.TS.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_145301960.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_145557475.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_145756449.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_145945633.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_150052971.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_150450946.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_150838898.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_151401807.TS_1fps.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_151809758.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_151904203.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_152205864.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_180723865.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_183038977.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_183236948.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_185700409.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_185745918.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_185825993.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_185915441.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_185926616.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_185935841.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_185947458.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_190031545.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_190119992.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_190421442.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_190810161.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_191035598.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_195741309.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_200743383.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_202334844.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_202346508.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_204038670.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_204621800.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Phantom 3 Professional - Container ship Maersk Laberinto_1fps.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/ROKS Lee Eokgi (SS 071) Transits in Close Formation During RIMPAC_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Royal Australian Navy Canberra Class Amphibious Ship HMAS Canberra (L02)_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Sailing Awayair2s_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Sterett and Chinese Frigate Steaming_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/U.S. Navy Aircraft Carrier Aerial - Flight Deck_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/UNITAS2025_Mayport_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/USS Bonhomme Richard transits under the Golden Gate Bridge_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/USSMason_RedSea_360_1fps.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/USS_Blue_Ridge_Aerial_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Yacht_Utopia_IV_at_Port_Miami_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam1-12_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam1-6_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam1-8_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam1-9_1fps.mp4/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam2-4_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam2-7_1fps 1.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam3-0_1fps 1.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam3-1_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam3-2_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam3-4_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam3-5_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/venice_sea_dawn1-3_1fps(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/31st MEU Search and Rescue Flight Operations_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Boat on the horizon 20250506 Norwegian Pearl_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Boat on the horizon 20250517 - GALATEA_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Boats on the horizon - Training Ship - Prolific_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Chinese Navy multi-role frigate Hengshui (572) - Forty Ships and Submarines Steam in Close Formation During RIMPAC_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Chinese_carrier_01052026_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Damen FCS3307_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Drone video analytics to assist search and rescue in flooding_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/GUARDIAN 1 DAMEN FCS3307_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Ike SAR Team Trains in Arabian Gulf_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Lurssen PV 80 offshore patrol vessel_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Marine Traffic 20250930 Auto Advance_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Marine Traffic off Beesands 20250813_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Offshore CN Made in Italy - Elettra Commander - High Speed Interceptor_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Phantom 3 - Grimaldi Lines Grande Argentina at Sunset_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/SEA AXE 51 offshore supple ship binnenkomst scheveningen_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Search and Rescue Exercise_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Space Coast Boating - Sebastian Inlet - August 28, 2020 - E006_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Space Coast Boating - Sebastian Inlet - Boat Watching_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/USS George H.W. Bush At Sea – Aerial View_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/USS Iwo Jima and USS Fort Lauderdale Strait Transit Exercise_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Unbelievable Epic Zoom Powers P900 Nikon_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Vehicles Carrier Silver Ray Mavic 2 Pro_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Vitesse Mark II Combat Interceptor Boat - RAF Boats - TNI AL - Indonesia Navy_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Watching a fast boat Challaborough Bay horizon_1fps/_annotations.coco.json',
]

val_dirs = [
    '/data/abe-data/cutlass-node-annotations-node-only/Bulker VITOSHA, Pilot Vessel, Lough Foyle Ferry & Fishing Boat - Great Day On The Foyle_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Drone flight 360 view of Cruise ship - Celestyal Discovery at Patmos 2025_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/LPV_Southcom_1fps.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_140051186.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_140114600.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_140155800.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_141845705.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_144505544.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PointReyes_ContainerShip_1fps.mp4 (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/USS Coronado LCS 4_1fps (classified)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam1-11_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam3-3_1fps.mp4(ARC)/_annotations.coco.json',
]

test_dirs = [
    '/data/abe-data/cutlass-node-annotations-node-only/8K video CAR Ferries crossing the Stormy Sea and moor at the Port of Dover_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Aerial View of Tug Boat Elk River Pushing a Barge Down Delaware River_1fps/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/DJI Mavic 2 Pro - FPV SALAR Fisheries Patrol Vessel_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/LPV_DVIDS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_135612528.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_143123491.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_173908102.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_190256699.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/PXL_20250924_200044230.TS_1fps.mp4(ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/Sailboat off in the distance_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/US Military Hovercraft LCAC & Assault Amphibious Vehicle Beach Landing Aerial View_1fps (ARC)/_annotations.coco.json',
    '/data/abe-data/cutlass-node-annotations-node-only/sea_cam1-10_1fps.mp4 (ARC)/_annotations.coco.json',
]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
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
                    # The radio of all image in train dataset < 7
                   # follow the original impl
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


# Helper to build dataset list
def build_dataset_list(ann_files, pipeline):
    datasets = []
    for ann in ann_files:
        datasets.append(
            dict(
                type='CocoDataset',
                ann_file=ann,
                img_prefix='', # JSONs contain absolute paths
                classes=classes,
                filter_empty_gt=False,
                pipeline=pipeline,
            )
        )
    return datasets

train_datasets = build_dataset_list(train_dirs, pipeline=train_pipeline)
val_datasets = build_dataset_list(val_dirs, pipeline=test_pipeline)
test_datasets = build_dataset_list(test_dirs, pipeline=test_pipeline)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type='ConcatDataset',
        datasets=train_datasets,
    ),
    val=dict(
        type='ConcatDataset',
        datasets=val_datasets,
    ),
    test=dict(
        type='ConcatDataset',
        datasets=test_datasets,
    ),
)


# -------------------------
# 4) Training Params
# -------------------------
# optimizer
optimizer = dict(
    type='AdamW',
    lr=4e-4,
    weight_decay=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }
    )
)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10])

runner = dict(type='EpochBasedRunner', max_epochs=10)

# Checkpoint
load_from = 'checkpoints/co_dino_5scale_vit_large_coco.pth'

# Logging
WANDB_PROJECT = 'co-detr-hydra'
WANDB_ENTITY  = 'cashel'
WANDB_RUNNAME = 'codetr-large-maritime-0128-nodes'

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
work_dir = '/data/cashel-data/models/codetr/maritime/maritime-0128-nodes'

# Eval
evaluation = dict(interval=2, metric='bbox', save_best='bbox_mAP', classwise=True)
checkpoint_config = dict(interval=2, save_last=True, max_keep_ckpts=2)

