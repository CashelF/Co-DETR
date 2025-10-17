#!/usr/bin/env python3
# Distributed Co-DETR (MMDetection v2) video inference — simple, per-GPU chunks.
import os, sys, json, time, argparse, math
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from mmdet.apis import init_detector, inference_detector


def parse_args():
    ap = argparse.ArgumentParser("Distributed Co-DETR video inference (MMDet v2)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="out_codetr.mp4", help="base output name (rank suffix added)")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (cuda recommended)")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=8, help="frames per step per GPU")
    ap.add_argument("--frame-stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--resize-short", type=int, default=0, help="short side resize (0 = none)")
    ap.add_argument("--save-json", default="", help="optional: base path for JSON (rank suffix added)")
    ap.add_argument("--concat", action="store_true", help="rank0 tries to ffmpeg-concat outputs at the end")
    return ap.parse_args()


def short_side_resize(img: np.ndarray, short_side: int) -> np.ndarray:
    if short_side <= 0: return img
    h, w = img.shape[:2]
    scale = short_side / float(min(h, w))
    if abs(scale - 1.0) < 1e-6: return img
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)


def get_class_names(model) -> List[str]:
    names = getattr(model, "CLASSES", None)
    return list(names) if names is not None else []


def unpack_v2(result):
    # MMDet v2 result: list per class with Nx5 [x1,y1,x2,y2,score]
    bboxes_all, labels_all = [], []
    for cls_id, arr in enumerate(result):
        if arr is None or len(arr) == 0: continue
        a = np.asarray(arr, dtype=np.float32)
        bboxes_all.append(a)
        labels_all.append(np.full((a.shape[0],), cls_id, dtype=np.int32))
    if not bboxes_all:
        return np.zeros((0,5), np.float32), np.zeros((0,), np.int32)
    return np.concatenate(bboxes_all, 0), np.concatenate(labels_all, 0)


def draw_dets(img: np.ndarray, dets: np.ndarray, labels: np.ndarray,
              class_names: List[str], score_thr: float = 0.3,
              font_scale: float = 0.5, thickness: int = 2, draw_text: bool = True):
    for i in range(dets.shape[0]):
        x1, y1, x2, y2, s = dets[i]
        if s < score_thr: continue
        c = int(labels[i])
        color = ((37*(c+1))%255, (17*(c+2))%255, (29*(c+3))%255)
        x1i, y1i, x2i, y2i = map(lambda v: int(max(v, 0)), (x1, y1, x2, y2))
        cv2.rectangle(img, (x1i, y1i), (x2i, y2i), color, thickness)
        if draw_text:
            label = class_names[c] if c < len(class_names) else f"id{c}"
            txt = f"{label} {s:.2f}"
            (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            ytxt = max(0, y1i - 3)
            cv2.rectangle(img, (x1i, ytxt - th - base), (x1i + tw + 2, ytxt + base), color, -1)
            cv2.putText(img, txt, (x1i + 1, ytxt), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (255,255,255), 1, cv2.LINE_AA)


def rank0_print(rank, *a, **k):
    if rank == 0: print(*a, **k)


def open_video_info(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return w, h, fps, n


def process_chunk(args, rank, world_size, model, class_names, out_size):
    """Each rank processes a contiguous [start, end) chunk (after frame_stride)."""
    # compute “effective” frame indices after stride so each rank gets contiguous block
    _, _, fps, total = open_video_info(args.video)

    # Build the list of frame indices we actually process
    if args.frame_stride <= 1:
        keep_idxs = list(range(total))
    else:
        keep_idxs = list(range(0, total, args.frame_stride))

    # split evenly among ranks
    per = math.ceil(len(keep_idxs) / world_size)
    start_i = rank * per
    end_i   = min(len(keep_idxs), (rank + 1) * per)
    my_idxs = keep_idxs[start_i:end_i]

    # open reader
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"[rank {rank}] failed to open video")

    # prepare writer (rank-specific file)
    out_h, out_w = out_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = add_rank_suffix(args.out, rank)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"[rank {rank}] failed to open writer: {out_path}")

    json_path = add_rank_suffix(args.save_json, rank) if args.save_json else ""
    preds_dump = []

    @torch.no_grad()
    def infer_one(bgr):
        return inference_detector(model, bgr)

    # batch loop over my frame indices
    B = max(1, int(args.batch_size))
    t0 = time.time()

    # We’ll read sequentially by seeking to the first frame of each mini-batch, then reading B frames.
    i = 0
    while i < len(my_idxs):
        batch_idxs = my_idxs[i:i+B]
        i += len(batch_idxs)

        # Seek to first
        first = batch_idxs[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)

        # Read B frames sequentially
        frames = []
        cur_frame_idx = first
        for _ in batch_idxs:
            ret, fr = cap.read()
            if not ret:
                break
            # if stride>1, but we're reading sequentially between selected idxs, we may overshoot;
            # we already pre-selected indices; just trust the order.
            frames.append(fr)
            cur_frame_idx += 1

        # Resize
        frames_resized = [short_side_resize(fr, args.resize_short) for fr in frames]

        # Infer per frame (v2 API)
        results = [infer_one(img) for img in frames_resized]
        unpacked = [unpack_v2(r) for r in results]

        # Draw/write/JSON
        for idx_val, img, (dets, labels) in zip(batch_idxs, frames_resized, unpacked):
            draw_dets(img, dets, labels, class_names, score_thr=args.score_thr)
            writer.write(img)

            if json_path:
                if dets.size:
                    keep = dets[:, 4] >= args.score_thr
                    det_keep = dets[keep].tolist()
                    lab_keep = labels[keep].tolist() if labels.size else []
                else:
                    det_keep, lab_keep = [], []
                preds_dump.append({
                    "frame_index": int(idx_val),
                    "width": int(img.shape[1]),
                    "height": int(img.shape[0]),
                    "detections": [
                        {
                            "bbox_xyxy": det_keep[k][:4],
                            "score": float(det_keep[k][4]),
                            "label": int(lab_keep[k]) if k < len(lab_keep) else 0
                        }
                        for k in range(len(det_keep))
                    ]
                })

        # progress
        done = min(i, len(my_idxs))
        if done % (B*10) == 0 or done == len(my_idxs):
            dt = time.time() - t0
            fps_rt = (done) / max(1e-6, dt)
            rank0_print(rank, f"[rank {rank}] {done}/{len(my_idxs)} ~{fps_rt:.2f} FPS")

    cap.release()
    writer.release()

    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(preds_dump, f)
        rank0_print(rank, f"[rank {rank}] wrote {json_path}")

    rank0_print(rank, f"[rank {rank}] wrote {out_path}")
    return out_path, json_path


def add_rank_suffix(path: str, rank: int) -> str:
    if not path: return path
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext or ''}"


def maybe_concat_with_ffmpeg(rank, world_size, base_out):
    if rank != 0: return
    try:
        import subprocess
        # build concat list
        parts = [add_rank_suffix(base_out, r) for r in range(world_size)]
        # write a ffmpeg concat file
        list_path = base_out + ".concat.txt"
        with open(list_path, "w") as f:
            for p in parts:
                f.write(f"file '{os.path.abspath(p)}'\n")
        merged = base_out.replace(".mp4", ".merged.mp4")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", merged]
        subprocess.run(cmd, check=True)
        print(f"[rank 0] merged -> {merged}")
    except Exception as e:
        print(f"[rank 0] concat failed (ok, you can do it later): {e}")


def maybe_merge_json(rank, world_size, base_json):
    if rank != 0 or not base_json: return
    try:
        merged = []
        for r in range(world_size):
            rp = add_rank_suffix(base_json, r)
            if os.path.isfile(rp):
                with open(rp, "r", encoding="utf-8") as f:
                    merged.extend(json.load(f))
        out_path = base_json.replace(".json", ".merged.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f)
        print(f"[rank 0] merged JSON -> {out_path}")
    except Exception as e:
        print(f"[rank 0] JSON merge failed: {e}")


def main():
    args = parse_args()

    # init distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() and args.device.startswith("cuda") else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # set device
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = f"cuda:{torch.cuda.current_device()}"
    else:
        device = "cpu"

    if rank == 0:
        # read video meta once and broadcast
        w, h, fps, total = open_video_info(args.video)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        out_frame = short_side_resize(dummy, args.resize_short)
        out_h, out_w = out_frame.shape[:2]
        payload = (w, h, fps, total, out_h, out_w)
    else:
        payload = (0,0,0.0,0,0,0)

    obj = [payload]
    dist.broadcast_object_list(obj, src=0)
    w, h, fps, total, out_h, out_w = obj[0]

    # load model per rank
    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=device)
    class_names = get_class_names(model)

    # process my chunk
    out_path, json_path = process_chunk(args, rank, world_size, model, class_names, (out_h, out_w))

    # barrier then optional merging on rank0
    dist.barrier()
    if args.concat:
        maybe_concat_with_ffmpeg(rank, world_size, args.out)
    if args.save_json:
        maybe_merge_json(rank, world_size, args.save_json)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# torchrun --nproc_per_node=8 tools/run_inference.py --config projects/configs/co_deformable_detr/gladius_r50_1x.py --checkpoint work_dir/gladius/epoch_12.pth --video police_chase.mp4 --out police_chase_out.mp4 --device cuda --score-thr 0.35 --batch-size 8 --frame-stride 1 --concat