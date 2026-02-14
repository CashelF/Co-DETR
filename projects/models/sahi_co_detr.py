import torch
import torch.nn.functional as F
import numpy as np
from mmdet.models.builder import DETECTORS
from .co_detr import CoDETR
from mmcv.ops import batched_nms

@DETECTORS.register_module()
class SahiCoDETR(CoDETR):
    def __init__(self, sahi_cfg=None, **kwargs):
        super(SahiCoDETR, self).__init__(**kwargs)
        if sahi_cfg is None:
            self.sahi_cfg = dict(
                crop_size=(480, 480),
                stride=320,
                target_size=(1024, 1024), 
                batch_size=4,
                global_score_thr=0.3, # Trust global less
                global_area_thr=32*32 # Discard global preds smaller than feature stride
            )
        self.sahi_cfg = sahi_cfg

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Two-Pass SAHI Inference with Batching and Scale-Aware Merging."""
        
        if isinstance(img, list): img = img[0]
        assert len(img) == 1, "SahiCoDETR only supports batch_size=1 per GPU"
             
        device = img.device
        h, w = img.shape[-2:]
        target_w, target_h = self.sahi_cfg['target_size']
        crop_h, crop_w = self.sahi_cfg['crop_size']
        
        # ---------------------------------------------------------
        # PASS 1: Global Context (The "Big Ship" Detector)
        # ---------------------------------------------------------
        img_global = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        img_meta_global = img_metas[0].copy()
        img_meta_global['img_shape'] = (target_h, target_w, 3)
        img_meta_global['scale_factor'] = np.array(
            [target_w / w, target_h / h, target_w / w, target_h / h], dtype=np.float32)

        # Force rescale=True so we get coords in original (H, W) space immediately
        results_global = super().simple_test(img_global, [img_meta_global], proposals, rescale=True)
        
        # ---------------------------------------------------------
        # SCALE FILTERING
        # ---------------------------------------------------------
        all_bboxes = []
        all_scores = []
        all_labels = []

        # Process Global Results
        # Global pass is BAD at small objects. Filter them out to prevent bad NMS later.
        global_area_thr = self.sahi_cfg.get('global_area_thr', 0)
        
        for cls_id, boxes in enumerate(results_global[0]):
            if len(boxes) == 0: continue
            
            # boxes is (N, 5) -> x1, y1, x2, y2, score
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = widths * heights
            
            # KEEP only if area is large enough (trusted global detection)
            keep_idxs = areas > global_area_thr
            
            if keep_idxs.any():
                valid_boxes = torch.from_numpy(boxes[keep_idxs]).to(device)
                all_bboxes.append(valid_boxes[:, :4])
                all_scores.append(valid_boxes[:, 4])
                all_labels.append(torch.full((valid_boxes.shape[0],), cls_id, device=device, dtype=torch.long))

        # ---------------------------------------------------------
        # PASS 2: Slicing
        # ---------------------------------------------------------
        stride = self.sahi_cfg.get('stride', int(crop_h * 0.75))
        batch_size = self.sahi_cfg.get('batch_size', 4)
        
        # 1. Prepare Slices (No inference yet)
        slice_patches = []
        slice_metas = []
        slice_origins = [] # (x1, y1) offsets

        y_steps = list(range(0, h - crop_h + 1, stride))
        if (h - crop_h) % stride != 0: y_steps.append(h - crop_h)
        x_steps = list(range(0, w - crop_w + 1, stride))
        if (w - crop_w) % stride != 0: x_steps.append(w - crop_w)

        for y1 in y_steps:
            for x1 in x_steps:
                # Create the patch
                patch = img[:, :, y1:y1+crop_h, x1:x1+crop_w]
                
                # OPTIMIZATION: Skip completely empty water patches if possible? 
                # (Skipping implemented here would require a quick variance check, ignored for now)

                # Zoom/Upscale patch
                patch_resized = F.interpolate(patch, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                # Meta needed for "rescale=True" to work
                meta = img_metas[0].copy()
                meta['ori_shape'] = (crop_h, crop_w, 3) # Model thinks this is the "original" size
                meta['img_shape'] = (target_h, target_w, 3)
                meta['scale_factor'] = np.array(
                    [target_w/crop_w, target_h/crop_h, target_w/crop_w, target_h/crop_h], dtype=np.float32)
                
                slice_patches.append(patch_resized)
                slice_metas.append(meta)
                slice_origins.append((x1, y1))

        # 2. Batched Inference Loop
        if len(slice_patches) > 0:
            for i in range(0, len(slice_patches), batch_size):
                # Stack batch
                batch_imgs = torch.cat(slice_patches[i : i+batch_size], dim=0) # (B, 3, H, W)
                batch_metas_subset = slice_metas[i : i+batch_size]
                batch_origins = slice_origins[i : i+batch_size]
                
                # Inference
                for j, single_img in enumerate(batch_imgs):
                    single_res = super().simple_test(single_img.unsqueeze(0), [batch_metas_subset[j]], proposals, rescale=True)
                    
                    origin_x, origin_y = batch_origins[j]
                    
                    for cls_id, boxes in enumerate(single_res[0]):
                        if len(boxes) == 0: continue
                        
                        # Boxes are already scaled to (crop_h, crop_w) because of rescale=True
                        # We just need to shift them
                        valid_boxes = torch.from_numpy(boxes).to(device)
                        
                        valid_boxes[:, 0] += origin_x
                        valid_boxes[:, 2] += origin_x
                        valid_boxes[:, 1] += origin_y
                        valid_boxes[:, 3] += origin_y
                        
                        all_bboxes.append(valid_boxes[:, :4])
                        all_scores.append(valid_boxes[:, 4])
                        all_labels.append(torch.full((valid_boxes.shape[0],), cls_id, device=device, dtype=torch.long))

        # ---------------------------------------------------------
        # MERGE & NMS
        # ---------------------------------------------------------
        if not all_bboxes:
            return results_global

        # Cat everything
        merged_bboxes = torch.cat(all_bboxes)
        merged_scores = torch.cat(all_scores)
        merged_labels = torch.cat(all_labels)

        # --- FIX: Robust NMS Config Extraction ---
        # Default NMS config
        nms_cfg = dict(type='nms', iou_threshold=0.5)
        
        # Check if test_cfg exists and handle Dict vs List
        if self.test_cfg is not None:
            if isinstance(self.test_cfg, dict):
                # Scenario A: It's a simple dict
                if 'nms' in self.test_cfg:
                    nms_cfg = self.test_cfg['nms']
                elif 'rcnn' in self.test_cfg and 'nms' in self.test_cfg['rcnn']:
                    # Scenario B: Nested inside 'rcnn'
                    nms_cfg = self.test_cfg['rcnn']['nms']
            
            elif isinstance(self.test_cfg, list):
                # Scenario C: It's a list
                for c in self.test_cfg:
                    if isinstance(c, dict):
                        if 'nms' in c:
                            nms_cfg = c['nms']
                            break
                        if 'rcnn' in c and 'nms' in c['rcnn']:
                            nms_cfg = c['rcnn']['nms']
                            break

        # Safety check: batched_nms requires a 'type'
        if 'type' not in nms_cfg:
            nms_cfg['type'] = 'nms'

        # Batched NMS (GPU accelerated, handles classes properly)
        dets, keep_indices = batched_nms(
            merged_bboxes, 
            merged_scores, 
            merged_labels, 
            nms_cfg
        )
        
        # Convert back to standard list[np.array] format for MMDetection
        final_results = [np.empty((0, 5), dtype=np.float32) for _ in range(len(results_global[0]))]
        
        if len(dets) > 0:
            dets = dets.cpu().numpy()
            labels = merged_labels[keep_indices].cpu().numpy()
            
            for i in range(len(dets)):
                cls_id = labels[i]
                final_results[cls_id] = np.vstack([final_results[cls_id], dets[i]])
            
        return [final_results]