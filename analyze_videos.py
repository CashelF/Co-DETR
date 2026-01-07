
import argparse
import os
import os.path as osp
import cv2
import mmcv
import torch
import numpy as np
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet.datasets import build_dataset
from mmdet.apis import init_detector
from pycocotools.cocoeval import COCOeval

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze videos: Calculate mAP per video and generate annotated videos')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('checkpoint', help='Checkpoint file path')
    parser.add_argument('--out-dir', default='analyzed_videos', help='Output directory for videos and results')
    parser.add_argument('--score-thr', type=float, default=0.3, help='BBox score threshold for visualization')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    return parser.parse_args()

def prepare_data(dataset, img_id):
    """Fetch and prepare data for a single image ID."""
    try:
        idx = dataset.img_ids.index(img_id)
    except ValueError:
        return None
    data = dataset[idx]
    # Collate
    data = collate([data], samples_per_gpu=1)
    return data

def results2json(dataset, results, img_ids):
    """Convert list of results to COCO JSON format."""
    json_results = []
    for img_id, result in zip(img_ids, results):
        for label, dets in enumerate(result):
            if dets is None or dets.size == 0:
                continue
            cat_id = dataset.cat_ids[label]
            for i in range(dets.shape[0]):
                score = float(dets[i, 4])
                bbox = dets[i, :4]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                json_results.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [bbox[0], bbox[1], width, height],
                    "score": score
                })
    return json_results

def evaluate_video(dataset, json_results, img_ids):
    """Calculate mAP for a specific set of images."""
    if not json_results:
        return 0.0
        
    cocoGt = dataset.coco
    # Filter GT for relevant images
    # Actually, simpler to just run COCOeval with specific imgIds
    # Converting json_results to a coco-like object
    cocoDt = cocoGt.loadRes(json_results)
    
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = img_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    # Return mAP @ IoU=0.5:0.95
    return cocoEval.stats[0]

def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    # Ensure test_mode is True for validation
    cfg.data.val.test_mode = True
    
    # Overwrite scale to standard COCO to save memory (ViT-L is heavy)
    print("Forcing image scale to (1333, 800) to avoid OOM...")
    for transform in cfg.data.val.pipeline:
        if transform['type'] == 'MultiScaleFlipAug':
            transform['img_scale'] = (1333, 800)
        elif transform['type'] == 'Resize':
            transform['img_scale'] = (1333, 800)
            
    dataset = build_dataset(cfg.data.val)
    print(f"Loaded dataset with {len(dataset)} images.")
    
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    # Group images by video_id
    print("Grouping images by video...")
    video_map = {} # video_id -> list of (frame_id, img_id)
    
    for i, img_id in enumerate(dataset.img_ids):
        # Efficient way: utilize data_infos if available
        info = dataset.data_infos[i]
        vid = info.get('video_id', 'unknown')
        frame = info.get('frame_id', 0)
        
        if vid not in video_map:
            video_map[vid] = []
        video_map[vid].append((frame, img_id, info))
        
    print(f"Found {len(video_map)} videos.")
    
    video_scores = {}
    
    for vid_id, frames in video_map.items():
        # Sort by frame_id to ensure temporal consistency
        frames.sort(key=lambda x: x[0])
        print(f"\nProcessing Video ID: {vid_id} ({len(frames)} frames)...")
        
        # Setup Video Writer
        # Get first image to determine size
        first_img_path = osp.join(dataset.img_prefix, frames[0][2]['file_name'])
        first_img = mmcv.imread(first_img_path)
        height, width, _ = first_img.shape
        
        video_name = f"video_{vid_id}.mp4"
        video_path = osp.join(args.out_dir, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(video_path, fourcc, 5.0, (width, height)) # 5 FPS guess
        
        vid_results = []
        vid_img_ids = []
        
        for frame_idx, (frame_num, img_id, info) in enumerate(frames):
            # 1. Prepare Data
            # Note: We must maintain sequential order for memory
            # The dataset[idx] pipeline should handle 'is_video_first' if configured correctly.
            # But the 'prepare_data' call above re-collates.
            # We need to ensure we run on GPU.
            
            data = prepare_data(dataset, img_id)
            if data is None: continue
            
            if next(model.parameters()).is_cuda:
                data = scatter(data, [torch.device(args.device)])[0]
            
            # 2. Inference
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
                
            vid_results.append(result[0])
            vid_img_ids.append(img_id)
            
            # 3. Visualize
            # show_result usually returns the image if out_file is None and show=False
            # But mmdet show_result might expect 'img' as path or array.
            # Let's pass the image path or array.
            img_path = osp.join(dataset.img_prefix, info['file_name'])
            
            # Using model.show_result
            # It internally calls imshow_det_bboxes.
            # We want the array back.
            viz_img = model.show_result(
                img_path,
                result[0],
                score_thr=args.score_thr,
                wait_time=0,
                show=False,
                thickness=2,
                font_scale=0.8,
                bbox_color='green',
                text_color='green'
            )
            
            # Resize if needed? No, assuming consistency.
            out_video.write(viz_img)
            
            if frame_idx % 10 == 0:
                print(f"  Processed frame {frame_idx}/{len(frames)}", end='\r')
        
        out_video.release()
        print(f"\n  Saved video to {video_path}")
        
        # 4. Calculate mAP for this video
        print(f"  Calculating mAP for Video {vid_id}...")
        json_res = results2json(dataset, vid_results, vid_img_ids)
        
        try:
            mAP = evaluate_video(dataset, json_res, vid_img_ids)
            video_scores[vid_id] = mAP
            print(f"  --> Video {vid_id} mAP: {mAP:.4f}")
        except Exception as e:
            print(f"  --> Failed to calculate mAP for Video {vid_id}: {e}")
            video_scores[vid_id] = -1.0

    # Summary
    print("\n===========================================")
    print("Video mAP Summary")
    print("===========================================")
    for vid, score in video_scores.items():
        print(f"Video {vid}: {score:.4f}")
    
    # Save scores to file
    with open(osp.join(args.out_dir, 'video_scores.txt'), 'w') as f:
        for vid, score in video_scores.items():
            f.write(f"Video {vid}: {score:.4f}\n")

if __name__ == '__main__':
    main()
