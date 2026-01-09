
import os
import torch
import json
import mmcv
import numpy as np
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import init_detector
from mmdet.core import eval_map
from terminaltables import AsciiTable

def verify_tracking():
    # 1. Config and Paths
    config_file = '/home/cash/Co-DETR/projects/configs/co_dino_vit/gladius_vit_large_co_dino.py'
    checkpoint_file = '/home/cash/Co-DETR/work_dir/cashel_test_cosine/latest.pth'
    ann_file = '/data/cashel-data/abes-gladius-data-vid/val/labels.json'
    target_img_path = 'val/images/Teal1_1fps_Teal1_1fps_0015.jpg' # Relative to root? we will match suffix
    
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
        
    # 2. Find target video
    print("Searching for target video...")
    target_video_id = None
    target_img_filename = os.path.basename(target_img_path)
    
    for img in coco_data['images']:
        if target_img_filename in img['file_name']:
            target_video_id = img['video_id']
            print(f"Found image {img['file_name']} in video_id {target_video_id}")
            break
            
    if target_video_id is None:
        print("Error: Target image not found in annotations.")
        return

    # Get all frames for this video
    video_imgs = [img for img in coco_data['images'] if img['video_id'] == target_video_id]
    video_imgs.sort(key=lambda x: x['frame_id'])
    print(f"Video {target_video_id} has {len(video_imgs)} frames.")
    
    # 3. Load Model
    print("Loading model...")
    cfg = Config.fromfile(config_file)
    # Force CPU because GPUs are full -> User requested CUDA
    device = 'cuda'
    model = init_detector(cfg, checkpoint_file, device=device)
    model.eval()
    
    
    # Co-DETR structure: query_head is the main DINO head
    if hasattr(model, 'query_head'):
        head = model.query_head
    elif hasattr(model, 'bbox_head'):
         # If bbox_head is list, assume first is not it? 
         # CoDETR usually has query_head.
         head = model.bbox_head
    else:
        raise AttributeError("Cannot find head")

    num_new = head.num_query
    num_track = getattr(head, 'num_track_queries', 0)
    print(f"Model config: Num New Queries: {num_new}, Num Track Queries: {num_track}")
    
    import inspect
    print(f"Head Class: {head.__class__}")
    print(f"Head File: {inspect.getfile(head.__class__)}")
    
    # 4. Inference Loop
    results_for_eval = []
    annotations_for_eval = []
    
    # Reset model state for new video
    if hasattr(head, '_prev_decoder_cache'):
        head._prev_decoder_cache = {}

    # Setup Video Writer
    import cv2
    from mmdet.core.visualization import imshow_det_bboxes
    
    video_out_file = 'tracking_demo.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None
    
    print(f"Opening video writer: {video_out_file}")

    for i, img_info in enumerate(video_imgs):
        frame_id = img_info['frame_id']
        file_name = img_info['file_name']
        img_path = os.path.join('/data/cashel-data/abes-gladius-data-vid/val/images', os.path.basename(file_name))
        
        # Read image
        img = mmcv.imread(img_path)
        
        # Initialize writer once we know size
        if video_writer is None:
            h, w, _ = img.shape
            # FPS = 10 for visualization?
            video_writer = cv2.VideoWriter(video_out_file, fourcc, 5, (w, h))

        # ... (Preprocessing)
        h, w, _ = img.shape
        img_res = mmcv.imresize(img, (1333, 800)) # Approx scale
        # Normalize
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_norm = mmcv.imnormalize(img_res, mean, std, to_rgb=True)
        
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        is_video_first = (i == 0)

        img_meta = {
            'img_shape': img_res.shape,
            'ori_shape': img.shape,
            'pad_shape': img_res.shape, # Assume no pad
            'filename': img_path,
            'scale_factor': np.array([img_res.shape[1]/w, img_res.shape[0]/h, img_res.shape[1]/w, img_res.shape[0]/h]),
            'batch_input_shape': img_res.shape[:2],
            'video_id': target_video_id,
            'frame_id': frame_id,
            'is_video_first': is_video_first
        }
        
        with torch.no_grad():
            features = model.extract_feat(img_tensor)
            outs = head(features, [img_meta])
            
            # outs is Tuple(cls_scores, bbox_preds)
            cls_scores = outs[0][-1] # Last decoder layer [B, Q, C]
            bbox_preds = outs[1][-1] # Last decoder layer [B, Q, 4]
            
            # ... (Analysis logging) ...
            
            # Get BBoxes for mAP
            bbox_list = head.get_bboxes(
                outs[0], outs[1], None, None, None, [img_meta], rescale=True)
            # bbox_list[0] is (bboxes, labels)
            det_bboxes, det_labels = bbox_list[0]
            
            # --- Visualization ---
            vis_img = img.copy()
            
            # Split detections into New vs Track
            # We need original indices to separate New vs Track
            # But get_bboxes returns filtered/NMS results usually.
            # wait, get_bboxes performs NMS. We lose the query index mapping.
            # However, for visualization, we can just hack it:
            # If we rely on get_bboxes, we don't know which query produced which box.
            
            # Alternative: Since we are debugging, let's just use the raw top-k scores we printed earlier?
            # Or assume the model returns them in order? No, NMS shuffles.
            
            # Since this is a specialized verification script, let's manually filter the raw outputs 
            # instead of using get_bboxes for visualization.
            # We already computed 'indices' which are the raw query indices > threshold.
            
            # Recover boxes for these indices
            # bbox_preds is [B, Q, 4] (unnormalized xywh usually, or sigmoid)
            # CO-DETR head outputs normalized xywh? Need to check.
            # Usually outputs are (cx, cy, w, h) normalized.
            
            # Let's use the 'indices' we found earlier to build visualization lists.
            track_bboxes = []
            track_labels = []
            new_bboxes = []
            new_labels = []
            
            h, w, _ = img.shape
            
            # For visualization, we need to get the raw scores and labels from the head output
            # and then filter by a score threshold.
            # This part assumes `max_scores`, `labels`, and `indices` are available from previous analysis logging.
            # If not, they need to be computed here.
            
            # Re-calculating for visualization purposes, assuming a score_thr of 0.3
            score_thr = 0.2
            
            # Get scores and labels for all queries
            scores = cls_scores.sigmoid() # [B, Q, C]
            max_scores, labels = scores[0].max(-1) # [Q], [Q]
            
            # Filter by score threshold
            valid_mask = max_scores >= score_thr
            indices = torch.nonzero(valid_mask, as_tuple=True)[0]
            
            for idx in indices:
                score = max_scores[idx].item()
                label = labels[idx].item()
                is_track = idx >= num_new
                
                # Get bbox stats
                raw_box = bbox_preds[0, idx] # cx, cy, w, h normalized
                cx, cy, bw, bh = raw_box.tolist()
                
                # Convert to xyxy absolute
                cx *= w
                cy *= h
                bw *= w
                bh *= h
                x1 = cx - bw/2
                y1 = cy - bh/2
                x2 = cx + bw/2
                y2 = cy + bh/2
                
                bbox_with_score = np.array([x1, y1, x2, y2, score])
                
                if is_track:
                    track_bboxes.append(bbox_with_score)
                    track_labels.append(label)
                else:
                    new_bboxes.append(bbox_with_score)
                    new_labels.append(label)

            # Draw NEW (Bright Pink/Magenta)
            # BGR: (255, 0, 255)
            if new_bboxes:
                vis_img = imshow_det_bboxes(
                    vis_img,
                    np.vstack(new_bboxes),
                    np.array(new_labels, dtype=np.int32),
                    class_names=model.CLASSES,
                    bbox_color=(255, 0, 255),
                    text_color=(255, 0, 255),
                    score_thr=0.0, # Already filtered
                    show=False
                )
            
            # Draw TRACK (Bright Green) - Draw second to overlay
            # BGR: (0, 255, 0)
            if track_bboxes:
                vis_img = imshow_det_bboxes(
                    vis_img,
                    np.vstack(track_bboxes),
                    np.array(track_labels, dtype=np.int32),
                    class_names=model.CLASSES,
                    bbox_color=(0, 255, 0),
                    text_color=(0, 255, 0),
                    score_thr=0.0, # Already filtered
                    show=False
                )
                
            # Draw Counters
            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            
            # New Counter (Pink)
            text_new = f"New: {len(new_bboxes)}"
            cv2.putText(vis_img, text_new, (20, 50), font, font_scale, (255, 0, 255), thickness)
            
            # Track Counter (Green)
            text_track = f"Track: {len(track_bboxes)}"
            cv2.putText(vis_img, text_track, (20, 90), font, font_scale, (0, 255, 0), thickness)

            video_writer.write(vis_img)
            
            # Convert to list for eval
            # Format: [ [bboxes_class0], [bboxes_class1], ... ]
            formatted_res = [det_bboxes[det_labels == c].cpu().numpy() for c in range(80)]
            
            # ... (GT Matching and Eval Prep)
            
            # Get matching GT
            frame_anns = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_info['id']]
            gt_bboxes = []
            gt_labels = []
            for ann in frame_anns:
                bbox = ann['bbox'] # xywh
                # Convert to xyxy
                x, y, w, h = bbox
                gt_bboxes.append([x, y, x+w, y+h])
                gt_labels.append(ann['category_id'] - 1)
            
            if gt_bboxes:
                annotations_for_eval.append({
                    'bboxes': np.array(gt_bboxes, dtype=np.float32),
                    'labels': np.array(gt_labels, dtype=np.int64)
                })
            else:
                 annotations_for_eval.append({
                    'bboxes': np.zeros((0, 4), dtype=np.float32),
                    'labels': np.zeros((0, ), dtype=np.int64)
                })
                
            # Inspect Cache
            if hasattr(head, '_prev_decoder_cache'):
                # Seq key?
                # We need to guess it or just print values
                if not head._prev_decoder_cache:
                    print(f"Frame {frame_id} [Cache] Empty")
                else:
                    for k, v in head._prev_decoder_cache.items():
                        n_track = v['valid_length']
                        active = (v['track_info'][:, 0] >= 0).sum().item()
                        print(f"Frame {frame_id} [Cache] Key: {k} | Tracks: {n_track} | Active: {active}")
    
    if video_writer:
        video_writer.release()
        print(f"Saved video to {video_out_file}")

            
    # 5. Calc mAP
    print("\nCalculating mAP for video...")
    mean_ap, _ = eval_map(
        results_for_eval,
        annotations_for_eval,
        iou_thr=0.5,
        dataset=None,
        nproc=1)
    
    print(f"\nVIDEO mAP (IoU=0.5): {mean_ap:.4f}")

if __name__ == "__main__":
    verify_tracking()
