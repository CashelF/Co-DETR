import torch
import torch.nn.functional as F
import numpy as np
from mmdet.models.builder import DETECTORS
from .co_detr import CoDETR
from mmdet.core import bbox_overlaps
from mmcv.ops import nms

@DETECTORS.register_module()
class SahiCoDETR(CoDETR):
    """Co-DETR wrapper for Scale-Adaptive SAHI Inference.
    
    Args:
        sahi_cfg (dict): Configuration for SAHI inference.
            Expected keys:
            - crop_size (tuple): (h, w) of the slice. Default: (480, 480).
            - overlap_ratio (float): Overlap ratio between slices. Default: 0.25 (unused, calculated from stride).
            - stride (int): Stride for slicing. Default: 320.
            - target_size (tuple): (w, h) input size for the model. Default: (1024, 1024).
            - batch_size (int): Batch size for slice inference. Default: 4.
            
    """
    def __init__(self, sahi_cfg=None, **kwargs):
        super(SahiCoDETR, self).__init__(**kwargs)
        if sahi_cfg is None:
            sahi_cfg = dict(
                crop_size=(480, 480),
                stride=320,
                target_size=(1024, 1024),
                batch_size=4
            )
        self.sahi_cfg = sahi_cfg

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Override simple_test to implement Two-Pass SAHI Inference."""
        
        # img is Tensor (1, C, H, W)
        # img_metas is list[dict]
        
        # Check if single image
        if isinstance(img, list):
             img = img[0]
             
        assert len(img) == 1, "SahiCoDETR only supports batch_size=1 (per device) for now"
             
        device = img.device
        h, w = img.shape[-2:]
        target_w, target_h = self.sahi_cfg['target_size']
        
        # --- PASS 1: Global Context ---
        # Resize full image to target_size
        img_global = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Update img_metas for global pass
        img_metas_global = [meta.copy() for meta in img_metas]
        for meta in img_metas_global:
            meta['img_shape'] = (target_h, target_w, 3)
            ori_h, ori_w = meta['ori_shape'][:2]
            meta['scale_factor'] = np.array([target_w / ori_w, target_h / ori_h, target_w / ori_w, target_h / ori_h], dtype=np.float32)

        # Run Global Inference
        results_global = super().simple_test(img_global, img_metas_global, proposals, rescale=True)
        
        # --- PASS 2: Slicing (Micro-Zoom) ---
        crop_h, crop_w = self.sahi_cfg['crop_size']
        stride = self.sahi_cfg.get('stride', int(crop_h * 0.75))
        batch_size = self.sahi_cfg.get('batch_size', 4)
        
        all_detections = [] # list of (bboxes, labels)
        
        # Collect global results first
        res_global = results_global[0] 
        for cls_idx, bboxes in enumerate(res_global):
            if len(bboxes) > 0:
                labels = np.full(len(bboxes), cls_idx, dtype=np.long)
                all_detections.append((bboxes, labels))
                
        # Generate Step Coordinates
        y_steps = range(0, h - crop_h + 1, stride)
        if (h - crop_h) % stride != 0:
            y_steps = list(y_steps) + [h - crop_h]
            
        x_steps = range(0, w - crop_w + 1, stride)
        if (w - crop_w) % stride != 0:
            x_steps = list(x_steps) + [w - crop_w]
            
        # Collect Slices
        slice_tensors = []
        slice_metas_list = []
        slice_coords = [] # (x1, y1)

        for y1 in y_steps:
            for x1 in x_steps:
                y2 = y1 + crop_h
                x2 = x1 + crop_w
                
                # Crop
                patch = img[:, :, y1:y2, x1:x2]
                
                # Resize to target_size (Zoom!)
                patch_resized = F.interpolate(patch, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                # Construct Patch Metas
                patch_meta = img_metas[0].copy()
                patch_meta['ori_shape'] = (crop_h, crop_w, 3) 
                patch_meta['img_shape'] = (target_h, target_w, 3)
                patch_meta['pad_shape'] = (target_h, target_w, 3)
                patch_meta['scale_factor'] = np.array([target_w / crop_w, target_h / crop_h, target_w / crop_w, target_h / crop_h], dtype=np.float32)
                
                slice_tensors.append(patch_resized)
                slice_metas_list.append(patch_meta)
                slice_coords.append((x1, y1))

        # Sequential Inference Loop
        for i, patch in enumerate(slice_tensors):
            patch_meta = slice_metas_list[i]
            x1, y1 = slice_coords[i]
            
            # Run Inference on single patch
            # Pass as list [patch] and [patch_meta] creates batch=1
            results_patch = super().simple_test(patch, [patch_meta], proposals, rescale=True)
            res_patch = results_patch[0]
            
            for cls_idx, bboxes in enumerate(res_patch):
                if len(bboxes) > 0:
                    # Shift boxes back to global coordinates
                    bboxes[:, 0] += x1
                    bboxes[:, 2] += x1
                    bboxes[:, 1] += y1
                    bboxes[:, 3] += y1
                    
                    labels = np.full(len(bboxes), cls_idx, dtype=np.long)
                    all_detections.append((bboxes, labels))

        # --- MERGE ---
        if not all_detections:
            return results_global
            
        final_results = [np.empty((0, 5), dtype=np.float32) for _ in range(len(res_global))]
        
        # Flatten all detections into a single list per class
        detections_per_class = [[] for _ in range(len(res_global))]
        
        for bboxes, labels in all_detections:
            for i in range(len(bboxes)):
                cls_id = labels[i]
                detections_per_class[cls_id].append(bboxes[i])
                
        # NMS Config
        iou_thr = 0.5
        nms_cfg = None
        if isinstance(self.test_cfg, list):
            for cfg in self.test_cfg:
                if 'nms' in cfg:
                    nms_cfg = cfg['nms']
                    break
        elif isinstance(self.test_cfg, dict):
            nms_cfg = self.test_cfg.get('nms')
            
        if nms_cfg is not None:
            iou_thr = nms_cfg.get('iou_threshold', 0.5)

        # Apply NMS
        for cls_idx, dets in enumerate(detections_per_class):
            if len(dets) == 0:
                continue
            dets = np.array(dets, dtype=np.float32)
            
            dets_tensor = torch.from_numpy(dets).to(device)
            keep_inds = nms(dets_tensor[:, :4].contiguous(), dets_tensor[:, 4].contiguous(), iou_thr)[1]
            
            final_results[cls_idx] = dets[keep_inds.cpu().numpy()]
            
        return [final_results]
