# Copyright (c) OpenMMLab. All rights reserved.
import copy
import numpy as np
from mmdet.datasets.builder import DATASETS, build_dataset
from mmdet.datasets.pipelines import Compose

@DATASETS.register_module()
class HybridMosaicDataset:
    """Hybrid dataset wrapper for Scale-Adaptive SAHI training.

    Args:
        dataset (:obj:`CustomDataset` | dict): The dataset to be wrapped.
        pipeline_a (list[dict]): Pipeline for Mode A (Global Context).
        pipeline_b (list[dict]): Pipeline for Mode B (Micro-Zoom Slicing).
        prob (float): Probability of choosing Mode B. Default: 0.5.
        crop_size (tuple): (h, w) for the crop in Mode B. Default: (480, 480).
        img_scale (tuple): (w, h) target size for both modes. Default: (1024, 1024).
    """

    def __init__(self, 
                 dataset, 
                 pipeline_a, 
                 pipeline_b, 
                 prob=0.5, 
                 crop_size=(480, 480),
                 img_scale=(1024, 1024)):
        
        if isinstance(dataset, dict):
            self.dataset = build_dataset(dataset)
        else:
            self.dataset = dataset
            
        self.pipeline_a = Compose(pipeline_a)
        self.pipeline_b = Compose(pipeline_b)
        self.prob = prob
        self.crop_size = crop_size
        self.img_scale = img_scale
        
        # Copy metadata from dataset
        self.CLASSES = self.dataset.CLASSES
        if hasattr(self.dataset, 'PALETTE'):
            self.PALETTE = self.dataset.PALETTE
        self.flag = self.dataset.flag if hasattr(self.dataset, 'flag') else np.zeros(len(self.dataset), dtype=np.uint8)

    def __len__(self):
        return len(self.dataset)

    def decide_mode(self, results):
        """Decide between Mode A (Global) and Mode B (Zoom) based on object sizes."""
        if 'gt_bboxes' not in results or len(results['gt_bboxes']) == 0:
            return np.random.rand() < self.prob

        bboxes = results['gt_bboxes']
        # Calculate areas (assuming [x1, y1, x2, y2] format)
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

        # Check for Giants (> 200,000 px^2) -> Force Mode A
        # preserve context for massive ships
        if np.any(areas > 200000):
            return False  # Mode A

        # Check for Micro-Only (All objects < 2,000 px^2) -> Force Mode B
        # tiny objects need zoom to be seen
        if np.all(areas < 2000):
            return True   # Mode B

        # Default: Random probability
        return np.random.rand() < self.prob

    def __getitem__(self, idx):
        # 1. Load basic info (Image + Annotations) from the underlying dataset
        # The underlying dataset should define a pipeline that only loads image and anns
        results = self.dataset[idx]
        
        # 2. Decide Mode
        use_mode_b = self.decide_mode(results)
        
        if use_mode_b:
            # Mode B: Micro-Zoom Slicing
            results = self._mode_b_transform(results)
        else:
            # Mode A: Global Context
            results = self._mode_a_transform(results)
            
        return results

    def _mode_a_transform(self, results):
        """Apply Global Context pipeline."""
        return self.pipeline_a(results)

    def _mode_b_transform(self, results):
        """Apply Micro-Zoom pipeline with Auto-Centering (Refined)."""
        img = results['img']
        if 'img_shape' not in results:
            results['img_shape'] = img.shape
            
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size
        
        # Ensure crop size is not larger than image
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)
        
        # --- 1. Determine Crop Coordinates ---
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            bboxes = results['gt_bboxes']
            # Pick a random box
            rand_idx = np.random.randint(len(bboxes))
            bbox = bboxes[rand_idx]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
        else:
            center_x = w // 2
            center_y = h // 2

        x1 = int(max(0, center_x - crop_w // 2))
        y1 = int(max(0, center_y - crop_h // 2))
        x2 = int(min(w, x1 + crop_w))
        y2 = int(min(h, y1 + crop_h))
        
        # Shift back if we hit the right/bottom edge
        if x2 == w: x1 = max(0, w - crop_w)
        if y2 == h: y1 = max(0, h - crop_h)

        patch = np.array([x1, y1, x2, y2])

        # --- 2. Filter BBoxes (Visibility Check) ---
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            gt_labels = results['gt_labels']
            
            # Calculate original areas
            original_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
            
            # Get intersection of boxes with the patch
            xx1 = np.maximum(gt_bboxes[:, 0], patch[0])
            yy1 = np.maximum(gt_bboxes[:, 1], patch[1])
            xx2 = np.minimum(gt_bboxes[:, 2], patch[2])
            yy2 = np.minimum(gt_bboxes[:, 3], patch[3])
            
            w_inter = np.maximum(0, xx2 - xx1)
            h_inter = np.maximum(0, yy2 - yy1)
            inter_areas = w_inter * h_inter
            
            # Check visibility ratio
            visibility = inter_areas / (original_areas + 1e-6)
            keep_mask = visibility >= 0.5
            
            # Also filter small boxes (e.g., < 1px)
            keep_mask = keep_mask & (w_inter >= 1) & (h_inter >= 1)
            
            # Apply filter
            gt_bboxes = gt_bboxes[keep_mask]
            gt_labels = gt_labels[keep_mask]
            
            # Use Intersection (Clipped) Coords
            xx1 = xx1[keep_mask]
            yy1 = yy1[keep_mask]
            xx2 = xx2[keep_mask]
            yy2 = yy2[keep_mask]
            
            # Stack into new bboxes
            gt_bboxes = np.stack([xx1, yy1, xx2, yy2], axis=1)

            # Transform boxes to patch coordinates
            gt_bboxes[:, [0, 2]] -= patch[0]
            gt_bboxes[:, [1, 3]] -= patch[1]
            
            results['gt_bboxes'] = gt_bboxes
            results['gt_labels'] = gt_labels

        # --- 3. Crop Image ---
        img_crop = img[y1:y2, x1:x2]
        results['img'] = img_crop
        results['img_shape'] = img_crop.shape
        # NOTE: We do NOT update 'ori_shape' usually, as that tracks the file on disk.
        # But for the subsequent `Resize`, it looks at `img_shape`.
        
        # Apply Pipeline B (Resize UP)
        return self.pipeline_b(results)
