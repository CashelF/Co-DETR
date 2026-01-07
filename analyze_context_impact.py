
import argparse
import copy
import os
import torch
import numpy as np
import random
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector
from mmdet.core import bbox2result

import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze temporal context impact')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--target-id', type=int, default=1801, help='Taget Image ID')
    parser.add_argument('--prev-id', type=int, default=1800, help='Previous frame Image ID')
    parser.add_argument('--video-id', type=int, default=8, help='Video ID of the sequence')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    return parser.parse_args()

def prepare_data(dataset, img_id, override_meta=None):
    """
    Fetch data info/pipeline for a specific image ID.
    override_meta: dict to update/force specific keys in img_metas[0]
                   e.g. {'is_video_first': True}
    """
    # 1. Find index in dataset
    # We need to map img_id to dataset index. 
    # CocoVideoDataset usually builds .data_infos list. 
    # We can search self.img_ids if available, or iterate.
    
    # Try efficient lookup if possible, else linear search
    try:
        idx = dataset.img_ids.index(img_id)
    except ValueError:
        print(f"Image ID {img_id} not found in dataset.")
        return None

    # 2. Get the data item
    data = dataset[idx]
    
    # 3. Override meta if needed
    if override_meta:
        # data['img_metas'] is a DataContainer. 
        # Its .data attribute holds the content (which is a list of lists/dicts depending on collate)
        # But here 'data' is a single sample from dataset[idx]. 
        # Usually dataset[idx]['img_metas'] is a DataContainer wrapping a single dictionary? 
        # Or just a DC wrapping [dict]?
        
        # Let's inspect the type safely.
        metas_container = data['img_metas']
        # For standard MMDet datasets: img_metas is DataContainer(cpu_only=True) wrapping a dict or list.
        # But in our pipeline we used DC(img_meta, cpu_only=True) -> wrapping a dict.
        
        if hasattr(metas_container, 'data'):
            # It is a DC.
            content = metas_container.data
            if isinstance(content, dict):
                 for k, v in override_meta.items():
                     content[k] = v
            elif isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
                 for k, v in override_meta.items():
                     content[0][k] = v
        else:
            # Fallback if it's raw list/dict (unlikely with mmdet pipeline)
             pass
            
    return data

def run_inference(model, data, device):
    # Collate and scatter
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified device
        from mmcv.parallel import scatter
        data = scatter(data, [device.index] if device.type == 'cuda' else None)[0]
    else:
        # cpu
        pass
    
    # Forward test
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    return results[0]

def evaluate_single_image(dataset, result, img_id):
    """
    Calculate mAP/score for a single image prediction against ground truth.
    Utilizes COCOeval logic for a single image.
    """
    # Convert result (list of arrays) to coco format list of dicts
    # result is [class0_dets, class1_dets, ...]
    
    json_results = []
    for cls_ind, dets in enumerate(result):
        cat_id = dataset.cat_ids[cls_ind]
        # Result from MMDetection is a list of numpy arrays, one arrays per class.
        # dets shape: (N, 5) => [x1, y1, x2, y2, score]
        
        if dets is None:
            continue
        if isinstance(dets, list): # Should be numpy
            dets = np.array(dets)
            
        if dets.size == 0:
            continue
            
        # Debug only if needed
        # print(f"Class {cls_ind} dets shape: {dets.shape}")

        for i in range(dets.shape[0]):
             det = dets[i]
             if det.shape[0] < 5:
                 print(f"Warning: Malformed detection at index {i} for class {cls_ind}: {det}")
                 continue
                 
             score = det[4]
             bbox = det[:4]
             # Convert xyxy to xywh
             bbox_xywh = [
                 bbox[0],
                 bbox[1],
                 bbox[2] - bbox[0],
                 bbox[3] - bbox[1]
             ]
             json_results.append({
                 "image_id": img_id,
                 "category_id": cat_id,
                 "bbox": bbox_xywh.tolist() if isinstance(bbox_xywh, np.ndarray) else bbox_xywh,
                 "score": float(score)
             })
            
    if not json_results:
        return 0.0, []

    # Creates a dummy coco object for just this image's ground truth
    # But simpler: use the main dataset.coco
    cocoGt = dataset.coco
    
    # Create cocoDt
    cocoDt = cocoGt.loadRes(json_results)
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = [img_id]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # Return mAP @ IoU=0.5:0.95 (index 0 in stats)
    return cocoEval.stats[0], json_results

def print_top_dets(dataset, json_results, top_k=5):
    # sort by score desc
    sorted_res = sorted(json_results, key=lambda x: x['score'], reverse=True)
    print(f"  Top {top_k} detections:")
    for i, res in enumerate(sorted_res[:top_k]):
        cat_name = dataset.CLASSES[dataset.cat2label[res['category_id']]]
        print(f"    {i+1}. {cat_name}: {res['score']:.4f} bbox={res['bbox']}")

def main():
    args = parse_args()
    
    # 1. Load config and model
    cfg = Config.fromfile(args.config)
    
    # Ensure test pipeline collects the necessary keys if not already present
    # (The user verified pipeline updates in previous step, so we trust it)
    
    dataset = build_dataset(cfg.data.val)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    
    target_id = args.target_id
    prev_id = args.prev_id
    
    print(f"Loaded dataset: {len(dataset)} samples.")
    print(f"Target Image ID: {target_id}")
    print(f"Previous Image ID: {prev_id}")
    
    # =======================================================
    # Condition 1: No Context
    # =======================================================
    print("\n-------------------------------------------------------------")
    print("Running Experiment: NO CONTEXT")
    print("-------------------------------------------------------------")
    
    # Reset model memory explicitly if possible? 
    # The head.py logic checks for `is_video_first`.
    
    # Prepare data forcing is_video_first=True
    data_no_context = prepare_data(dataset, target_id, override_meta={'is_video_first': True})
    
    # Run
    # Note: init_detector wraps in MMDataParallel? No, usually just the model.
    # But run_inference handles collation.
    
    res_no_context = run_inference(model, data_no_context, torch.device(args.device))
    
    ap_no_context, dets_no_context = evaluate_single_image(dataset, res_no_context, target_id)
    print(f"--> No Context mAP: {ap_no_context:.4f}")
    print_top_dets(dataset, dets_no_context)


    # =======================================================
    # Condition 2: True Context
    # =======================================================
    print("\n-------------------------------------------------------------")
    print("Running Experiment: TRUE CONTEXT")
    print("-------------------------------------------------------------")
    
    # Strategy:
    # 1. Run prev_id with is_video_first=True (to flush any old memory and start this sequence)
    # 2. Run target_id with is_video_first=False
    
    # Step A: Warmup with prev frame
    data_prev = prepare_data(dataset, prev_id, override_meta={'is_video_first': True})
    _ = run_inference(model, data_prev, torch.device(args.device))
    
    # Step B: Target
    # Ensure we link it to the same video/sequence implicitly by NOT resetting
    # The head logic checks if should reset. We want it NOT to reset.
    # video_id, frame_id are correct in dataset.
    data_true_context = prepare_data(dataset, target_id, override_meta={'is_video_first': False})
    
    res_true_context = run_inference(model, data_true_context, torch.device(args.device))
    
    ap_true_context, dets_true_context = evaluate_single_image(dataset, res_true_context, target_id)
    print(f"--> True Context mAP: {ap_true_context:.4f}")
    print_top_dets(dataset, dets_true_context)

    # Visualize results
    print("Saving True Context visualization to 'true_context_detections.jpg'...")
    # Extract filename from data_true_context
    try:
        metas = data_true_context['img_metas']
        img_metas = None
        
        if isinstance(metas, list):
            # If list, check first element
            if len(metas) > 0:
                first = metas[0]
                if hasattr(first, 'data'):
                    img_metas = first.data
                else:
                    img_metas = first
        elif hasattr(metas, 'data'):
            img_metas = metas.data
            
        if isinstance(img_metas, list):
            img_metas = img_metas[0]
            
        if img_metas and 'filename' in img_metas:
            filename = img_metas['filename']
            model.show_result(filename, res_true_context, out_file='true_context_detections.jpg')
            print(f"Successfully saved true_context_detections.jpg for {filename}")
        else:
            print("Could not find filename in img_metas")
            
    except Exception as e:
        print(f"Failed to visualize results: {e}")


    # =======================================================
    # Condition 3: Random Context
    # =======================================================
    print("\n-------------------------------------------------------------")
    print("Running Experiment: RANDOM CONTEXT")
    print("-------------------------------------------------------------")
    
    # Strategy:
    # 1. Pick a random image that is NOT the prev or target.
    # 2. Hack its meta to look like the 'previous' frame (same video_id, frame_id - 1) 
    #    so the model accepts it as temporal context.
    #    BUT wait, the model checks video_id match?
    #    If we just run it as "is_video_first=True", it starts a sequence.
    #    Then we run target. The target matches the video_id/frame_id logic.
    #    So we need the random image to pretend to be part of video=8, frame=54.
    
    random_idx = random.randint(0, len(dataset)-1)
    # Ensure not target or prev
    while dataset.img_ids[random_idx] in [target_id, prev_id]:
         random_idx = random.randint(0, len(dataset)-1)
         
    random_img_id = dataset.img_ids[random_idx]
    print(f"Selected Random Context Image ID: {random_img_id}")
    
    # 1. Run random image as START of sequence
    # BUT we need to spoof its video_id to match the target, otherwise the head might detect video change and reset?
    # Actually checking head implementation:
    # _get_sequence_key uses video_id.
    # If we run Random Image (Video X) then Target (Video 8), key changes -> Reset.
    # So we MUST spoof video_id of random image to be 8.
    
    data_random = prepare_data(dataset, random_img_id, override_meta={
        'is_video_first': True,
        'video_id': args.video_id,   # Spoof as target video
        'frame_id': args.prev_id - 100 # Just an earlier frame? Or exactly prev_id?
        # Let's spoof it exactly as the previous frame to simulate "corrupted previous frame"
    })
    # Safe access to modify frame_id
    metas_container = data_random['img_metas']
    if hasattr(metas_container, 'data'):
        # DC
        if isinstance(metas_container.data, dict):
             metas_container.data['frame_id'] = 54
        elif isinstance(metas_container.data, list) and len(metas_container.data) > 0:
             metas_container.data[0]['frame_id'] = 54
    elif isinstance(metas_container, list) and len(metas_container) > 0:
        # Could be raw list or list of DC
        item = metas_container[0]
        if hasattr(item, 'data'):
             # It is a DC inside a list
             if isinstance(item.data, dict):
                 item.data['frame_id'] = 54
             elif isinstance(item.data, list) and len(item.data) > 0:
                 item.data[0]['frame_id'] = 54
        else:
             # Raw list
             metas_container[0]['frame_id'] = 54
    
    _ = run_inference(model, data_random, torch.device(args.device))
    
    # 2. Run Target
    data_rand_context_target = prepare_data(dataset, target_id, override_meta={'is_video_first': False})
    
    res_rand_context = run_inference(model, data_rand_context_target, torch.device(args.device))
    
    ap_rand_context, dets_rand_context = evaluate_single_image(dataset, res_rand_context, target_id)
    print(f"--> Random Context mAP: {ap_rand_context:.4f}")
    print_top_dets(dataset, dets_rand_context)

    # Summary
    print("\n=============================================================")
    print("SUMMARY")
    print("=============================================================")
    print(f"No Context mAP:      {ap_no_context:.4f}")
    print(f"True Context mAP:    {ap_true_context:.4f}")
    print(f"Random Context mAP:  {ap_rand_context:.4f}")
    print("=============================================================")

if __name__ == '__main__':
    main()
