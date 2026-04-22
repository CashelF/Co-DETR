#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MMDetection bbox JSON results into a COCO-like teacher supervision file with scores."
    )
    parser.add_argument("--source-ann", required=True, help="Source COCO-style annotations used for inference")
    parser.add_argument("--bbox-json", required=True, help="MMDetection bbox JSON produced by --format-only")
    parser.add_argument("--output", required=True, help="Output COCO-like teacher supervision JSON")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Minimum score to keep")
    parser.add_argument("--max-dets-per-image", type=int, default=None, help="Optional cap on detections kept per image")
    return parser.parse_args()


def main():
    args = parse_args()
    source_ann_path = Path(args.source_ann)
    bbox_json_path = Path(args.bbox_json)
    output_path = Path(args.output)

    with source_ann_path.open() as f:
        source = json.load(f)
    with bbox_json_path.open() as f:
        detections = json.load(f)

    images = source.get("images", [])
    categories = source.get("categories", [])
    image_ids = {img["id"] for img in images}

    grouped = defaultdict(list)
    for det in detections:
        image_id = det.get("image_id")
        if image_id not in image_ids:
            continue
        score = float(det.get("score", 1.0))
        if score < args.score_threshold:
            continue
        grouped[image_id].append(det)

    annotations = []
    next_annotation_id = 1
    for image in images:
        image_id = image["id"]
        image_detections = grouped.get(image_id, [])
        image_detections.sort(key=lambda det: float(det.get("score", 1.0)), reverse=True)
        if args.max_dets_per_image is not None and args.max_dets_per_image > 0:
            image_detections = image_detections[:args.max_dets_per_image]

        for det in image_detections:
            bbox = det["bbox"]
            annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": det["category_id"],
                    "bbox": bbox,
                    "area": float(bbox[2]) * float(bbox[3]),
                    "iscrowd": 0,
                    "score": float(det.get("score", 1.0)),
                    "segmentation": [],
                }
            )
            next_annotation_id += 1

    teacher_coco = {
        "info": source.get("info", {"description": "Teacher supervision", "version": "1.0"}),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    for key in ("licenses", "ontology", "ontology_type"):
        if key in source:
            teacher_coco[key] = source[key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(teacher_coco, f)

    print(f"Wrote {len(annotations)} teacher annotations for {len(images)} images to {output_path}")


if __name__ == "__main__":
    main()
