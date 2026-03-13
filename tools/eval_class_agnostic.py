import sys
import os
import json
import pickle
import numpy as np

def main():
    pkl_file = 'work_dir/maritime_0128_nodes/test_results.pkl'
    
    # Notice: Co-DETR in projects configs runs dist_test.sh on test set by default.
    val_json_path = '/data/cashel-data/maritime-0128/test/labels.json'
    modified_json_path = '/data/cashel-data/maritime-0128/test/labels_binary.tmp.json'
    
    print("Loading predictions...")
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded {len(results)} image predictions.")

    print("Merging predictions to a single class...")
    new_results = []
    for pred in results:
        # pred is a list of N_c arrays of shape (K, 5), or tuple if masks are present
        bboxes = pred
        if isinstance(pred, tuple):
            bboxes = pred[0]
            
        if len(bboxes) > 0:
            merged = np.concatenate(bboxes, axis=0)
            # sort by score (column index 4) descending
            merged = merged[merged[:, 4].argsort()[::-1]]
        else:
            merged = np.zeros((0, 5), dtype=np.float32)
        new_results.append([merged])

    print("Loading gt json...")
    with open(val_json_path, 'r') as f:
        gt_data = json.load(f)

    print("Modifying gt json to a single class...")
    # Change all categories to a single category (id=1)
    gt_data['categories'] = [{'id': 1, 'name': 'object', 'supercategory': 'object'}]
    
    for ann in gt_data['annotations']:
        ann['category_id'] = 1
        
    print("Saving modified gt json...")
    with open(modified_json_path, 'w') as f:
        json.dump(gt_data, f)

    print("Running mmdetection evaluation...")
    from mmdet.datasets.coco import CocoDataset
    eval_dataset = CocoDataset(
        ann_file=modified_json_path,
        classes=('object',),
        pipeline=[], # Not needed for evaluation
        test_mode=True,
        img_prefix=''
    )
    metric = eval_dataset.evaluate(new_results, metric='bbox', classwise=True)
    print("Class-agnostic mAP evaluation complete:")
    for key, value in metric.items():
        print(f"{key}: {value}")
    
if __name__ == "__main__":
    main()
