#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_codetr_cache')

import mmcv
import numpy as np
import torch
from PIL import Image
from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.utils import get_logger
from mmdet.models import build_detector
from mmdet.utils import compat_cfg, replace_cfg_vals, setup_multi_processes, update_data_root

from projects import *  # noqa: F401,F403

try:
    from kd_replay.replay import KD_REPLAY_INDEX_KIND, apply_replay_recipe_to_image
except Exception:
    KD_REPLAY_INDEX_KIND = "replay_manifest"
    apply_replay_recipe_to_image = None


PACKET_FORMAT_VERSION = 1
DEFAULT_PAD_SIZE_DIVISOR = 32


def torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


@dataclass
class PacketEntry:
    shard: str
    offset: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export offline KD packets from Co-DETR exact-view shards.')
    parser.add_argument('--config', required=True, help='Co-DETR config file.')
    parser.add_argument('--checkpoint', required=True, help='Teacher checkpoint path.')
    parser.add_argument('--view-index', required=True, help='Path to the exact-view index JSON.')
    parser.add_argument('--output-dir', required=True, help='Directory to write packet shards and packet index.')
    parser.add_argument('--device', default='cuda', help='Inference device.')
    parser.add_argument('--batch-size', type=int, default=2, help='Teacher inference batch size.')
    parser.add_argument('--random-queries', type=int, default=300, help='Number of random reference points to distill.')
    parser.add_argument('--max-shards', type=int, default=None, help='Optional shard cap for smoke tests.')
    parser.add_argument('--worker-rank', type=int, default=0, help='Worker rank for shard-partitioned multi-GPU export.')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of shard-partitioned workers.')
    parser.add_argument('--index-out', default=None, help='Optional output path for this worker partial packet index.')
    return parser.parse_args()


def load_view_index(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_cfg(config_path: str) -> Config:
    cfg = Config.fromfile(config_path)
    cfg = replace_cfg_vals(cfg)
    update_data_root(cfg)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if 'pretrained' in cfg.model:
        cfg.model.pretrained = None
    elif 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if cfg.model.get('neck'):
        if isinstance(cfg.model.neck, list):
            for neck_cfg in cfg.model.neck:
                if neck_cfg.get('rfp_backbone') and neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
        elif cfg.model.neck.get('rfp_backbone') and cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

    cfg.model.train_cfg = None
    return cfg


def quiet_openmmlab_logging() -> None:
    for name in ('mmcv', 'mmdet', 'mmseg', 'mmengine'):
        get_logger(name, log_level=logging.ERROR)
        logging.getLogger(name).setLevel(logging.ERROR)


def get_silent_checkpoint_logger() -> logging.Logger:
    logger = logging.getLogger('codetr_export.checkpoint')
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.ERROR)
    return logger


def configure_runtime_threads() -> None:
    torch_threads = max(1, int(os.environ.get('KD_EXPORT_TORCH_THREADS', '1')))
    torch_interop_threads = max(1, int(os.environ.get('KD_EXPORT_TORCH_INTEROP_THREADS', '1')))
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(torch_threads)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(torch_interop_threads)
    except RuntimeError:
        pass


def build_model_from_cfg(cfg: Config, checkpoint: str, device: str):
    quiet_openmmlab_logging()
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(
        model,
        checkpoint,
        map_location='cpu',
        logger=get_silent_checkpoint_logger(),
    )
    model.to(device)
    model.eval()
    return model


def get_norm_stats(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    norm_cfg = cfg.get('img_norm_cfg', None)
    if norm_cfg is None:
        raise ValueError('Expected img_norm_cfg in the teacher config')
    mean = torch.tensor(norm_cfg['mean'], dtype=torch.float32) / 255.0
    std = torch.tensor(norm_cfg['std'], dtype=torch.float32) / 255.0
    return mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)


def find_pad_size_divisor(cfg: Config) -> int:
    candidates = []
    for key in ('train_pipeline', 'test_pipeline'):
        if key in cfg:
            candidates.append(cfg[key])
    if 'data' in cfg:
        for split in ('train', 'val', 'test'):
            split_cfg = cfg.data.get(split)
            if isinstance(split_cfg, dict) and 'pipeline' in split_cfg:
                candidates.append(split_cfg['pipeline'])

    def walk(transforms):
        for transform in transforms or []:
            if isinstance(transform, dict):
                if transform.get('type') == 'Pad' and 'size_divisor' in transform:
                    return int(transform['size_divisor'])
                if 'transforms' in transform:
                    found = walk(transform['transforms'])
                    if found is not None:
                        return found
        return None

    for pipeline in candidates:
        found = walk(pipeline)
        if found is not None:
            return found
    return DEFAULT_PAD_SIZE_DIVISOR


def round_up(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def collate_views(records: list[dict[str, Any]], mean: torch.Tensor, std: torch.Tensor, pad_size_divisor: int, device: torch.device):
    images = [record['image'].to(device=device, dtype=torch.float32) for record in records]
    heights = [int(image.shape[1]) for image in images]
    widths = [int(image.shape[2]) for image in images]
    pad_h = round_up(max(heights), pad_size_divisor)
    pad_w = round_up(max(widths), pad_size_divisor)

    batch = torch.zeros((len(images), 3, pad_h, pad_w), dtype=torch.float32, device=device)
    img_metas = []
    for idx, image in enumerate(images):
        height, width = heights[idx], widths[idx]
        normalized = (image.unsqueeze(0) - mean) / std
        batch[idx, :, :height, :width] = normalized[0]
        img_metas.append({
            'img_shape': (height, width, 3),
            'ori_shape': (height, width, 3),
            'pad_shape': (pad_h, pad_w, 3),
            'batch_input_shape': (pad_h, pad_w),
            'scale_factor': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            'flip': False,
            'flip_direction': None,
        })
    return batch, img_metas


def materialize_replay_records(records: list[dict[str, Any]]):
    if apply_replay_recipe_to_image is None:
        raise RuntimeError(
            'Replay-manifest teacher export requires kd_replay on PYTHONPATH. '
            'Set PYTHONPATH=/home/cash/kd-detr:${PYTHONPATH} before launching export.'
        )

    materialized = []
    for record in records:
        pil_image = Image.open(record['image_path']).convert('RGB')
        tensor = apply_replay_recipe_to_image(pil_image, record['recipe'], normalize=False)
        materialized.append({
            'distillation_packet_id': torch.tensor([int(record['distillation_packet_id'])], dtype=torch.int64),
            'image': tensor,
        })
    return materialized


def sample_random_points(packet_ids: list[int], num_queries: int, device: torch.device) -> torch.Tensor:
    points = []
    for packet_id in packet_ids:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(int(packet_id) % (2**31 - 1))
        unsigmoid = torch.randn((num_queries, 4), generator=generator, dtype=torch.float32)
        points.append(unsigmoid.sigmoid())
    return torch.stack(points, dim=0).to(device)


def build_aux_targets(random_points: torch.Tensor, embed_dims: int):
    batch_size, num_queries, _ = random_points.shape
    device = random_points.device
    aux_labels = torch.zeros((batch_size, num_queries), dtype=torch.long, device=device)
    aux_targets = random_points.detach().clone()
    aux_label_weights = torch.ones((batch_size, num_queries), dtype=torch.float32, device=device)
    aux_bbox_weights = torch.ones((batch_size, num_queries, 4), dtype=torch.float32, device=device)
    aux_feats = torch.zeros((batch_size, num_queries, embed_dims), dtype=torch.float32, device=device)
    attn_masks = None
    return (random_points, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks)


def iter_chunks(records: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start:start + batch_size]


def write_packet_index(index_path: Path, entries: dict[int, PacketEntry]) -> None:
    payload = {
        'format_version': PACKET_FORMAT_VERSION,
        'packets': {str(packet_id): {'shard': entry.shard, 'offset': entry.offset} for packet_id, entry in entries.items()},
    }
    with index_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> int:
    args = parse_args()
    configure_runtime_threads()
    view_index_path = Path(args.view_index).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.config)
    device = torch.device(args.device)
    model = build_model_from_cfg(cfg, args.checkpoint, args.device)
    mean, std = get_norm_stats(cfg)
    mean = mean.to(device)
    std = std.to(device)
    pad_size_divisor = find_pad_size_divisor(cfg)

    payload = load_view_index(view_index_path)
    index_kind = payload.get('kind')
    view_root = view_index_path.parent
    shard_names = sorted({entry['shard'] for entry in payload['views'].values()})
    if args.max_shards is not None:
        shard_names = shard_names[:args.max_shards]
    shard_names = shard_names[args.worker_rank::args.num_workers]

    packet_entries: dict[int, PacketEntry] = {}

    with torch.no_grad():
        for local_shard_idx, shard_name in enumerate(shard_names):
            view_records = torch_load_compat(view_root / shard_name)
            total_records = len(view_records)
            total_batches = (total_records + args.batch_size - 1) // args.batch_size
            print(
                f'rank {args.worker_rank}: start {shard_name} '
                f'({total_records} views, {total_batches} batches)',
                flush=True,
            )
            teacher_points_batches = []
            teacher_logits_batches = []
            teacher_boxes_batches = []
            random_points_batches = []
            random_logits_batches = []
            random_boxes_batches = []
            packet_ids = []

            for batch_idx, batch_records in enumerate(iter_chunks(view_records, args.batch_size), start=1):
                if index_kind == KD_REPLAY_INDEX_KIND:
                    batch_records = materialize_replay_records(batch_records)
                packet_ids_batch = [int(record['distillation_packet_id'].item()) for record in batch_records]
                images, img_metas = collate_views(batch_records, mean, std, pad_size_divisor, device)
                features = model.extract_feat(images)
                all_cls_scores, all_bbox_preds, _, topk_anchor, _ = model.query_head(features, img_metas)

                random_points = sample_random_points(packet_ids_batch, args.random_queries, device)
                aux_targets = build_aux_targets(random_points, model.query_head.embed_dims)
                random_cls_scores, random_bbox_preds, _, _ = model.query_head.forward_aux(features, img_metas, aux_targets, head_idx=0)

                packet_ids.extend(packet_ids_batch)
                teacher_points_batches.append(topk_anchor.detach().cpu().to(torch.float16).numpy())
                teacher_logits_batches.append(all_cls_scores[-1].detach().cpu().to(torch.float16).numpy())
                teacher_boxes_batches.append(all_bbox_preds[-1].detach().cpu().to(torch.float16).numpy())
                random_points_batches.append(random_points.detach().cpu().to(torch.float16).numpy())
                random_logits_batches.append(random_cls_scores[-1].detach().cpu().to(torch.float16).numpy())
                random_boxes_batches.append(random_bbox_preds[-1].detach().cpu().to(torch.float16).numpy())

                if batch_idx == 1 or batch_idx == total_batches or batch_idx % 4 == 0:
                    print(
                        f'rank {args.worker_rank}: {shard_name} '
                        f'batch {batch_idx}/{total_batches} '
                        f'({len(packet_ids)}/{total_records} views)',
                        flush=True,
                    )

            if args.num_workers > 1:
                packet_shard_name = f'packets-r{args.worker_rank:02d}-{local_shard_idx:06d}.npz'
            else:
                packet_shard_name = f'packets-{local_shard_idx:06d}.npz'
            np.savez_compressed(
                output_dir / packet_shard_name,
                packet_ids=np.asarray(packet_ids, dtype=np.int64),
                teacher_points=np.concatenate(teacher_points_batches, axis=0),
                teacher_logits=np.concatenate(teacher_logits_batches, axis=0),
                teacher_boxes=np.concatenate(teacher_boxes_batches, axis=0),
                random_points=np.concatenate(random_points_batches, axis=0),
                random_logits=np.concatenate(random_logits_batches, axis=0),
                random_boxes=np.concatenate(random_boxes_batches, axis=0),
            )
            for offset, packet_id in enumerate(packet_ids):
                packet_entries[int(packet_id)] = PacketEntry(shard=packet_shard_name, offset=offset)
            print(f'exported {len(packet_ids)} packets from {shard_name} -> {packet_shard_name}', flush=True)

    if args.index_out:
        index_path = Path(args.index_out).resolve()
    elif args.num_workers > 1:
        index_path = output_dir / f'packets.index.rank{args.worker_rank:02d}.json'
    else:
        index_path = output_dir / 'packets.index.json'
    write_packet_index(index_path, packet_entries)
    print(f'wrote {len(packet_entries)} packets to {index_path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
