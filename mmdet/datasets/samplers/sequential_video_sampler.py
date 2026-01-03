"""Sequential video samplers used for training."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler as _DistributedSampler

from mmdet.core.utils import sync_random_seed
from mmdet.utils import get_device


def _group_indices_by_video(dataset) -> List[List[int]]:
    """Group dataset indices by ``video_id``.

    Args:
        dataset: Dataset exposing ``data_infos`` entries with a ``video_id``
            key ordered by ``frame_id`` within each video.

    Returns:
        List[List[int]]: Dataset indices grouped by video in temporal order.
    """

    if not hasattr(dataset, 'data_infos'):
        raise AttributeError('Sequential video samplers require the dataset '
                             'to expose a data_infos attribute.')

    video_indices: List[List[int]] = []
    current: List[int] = []
    current_video_id = None

    for idx, info in enumerate(dataset.data_infos):
        video_id = info.get('video_id')
        if video_id is None:
            raise KeyError('Each data_info must include a "video_id" key to '
                           'enable sequential video sampling.')

        if current_video_id is None or video_id == current_video_id:
            current.append(idx)
            current_video_id = video_id
        else:
            video_indices.append(current)
            current = [idx]
            current_video_id = video_id

    if current:
        video_indices.append(current)

    return video_indices


class SequentialVideoSampler(Sampler[int]):
    """Sample frames sequentially within each video while shuffling videos."""

    def __init__(self, dataset, shuffle_videos: bool = True, seed: int | None = None):
        super().__init__(dataset)
        self.dataset = dataset
        self.shuffle_videos = shuffle_videos
        self.seed = seed
        self.video_indices = _group_indices_by_video(dataset)
        self.num_samples = len(dataset)
        self._epoch = 0

    def __iter__(self) -> Iterable[int]:
        if self.shuffle_videos:
            generator = torch.Generator()
            if self.seed is None:
                generator.seed()
            else:
                generator.manual_seed(self.seed + self._epoch)
            order = torch.randperm(
                len(self.video_indices), generator=generator).tolist()
        else:
            order = list(range(len(self.video_indices)))

        for vid_idx in order:
            for idx in self.video_indices[vid_idx]:
                yield idx

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


class DistributedSequentialVideoSampler(_DistributedSampler):
    """Distributed sampler that preserves the frame order within a video."""

    def __init__(self,
                 dataset,
                 num_replicas: int | None = None,
                 rank: int | None = None,
                 shuffle_videos: bool = True,
                 seed: int = 0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
        if not shuffle_videos:
            raise ValueError('DistributedSequentialVideoSampler requires '
                             'shuffle_videos=True to avoid deterministic '
                             'ordering every epoch.')

        device = get_device()
        self.seed = sync_random_seed(seed, device)
        self.video_indices = _group_indices_by_video(dataset)
        if len(self.video_indices) < self.num_replicas:
            raise ValueError('The dataset must contain at least as many videos '
                             'as replicas to use DistributedSequentialVideoSampler.')

        self._epoch = 0
        self._rank_indices: List[List[int]] = []
        self._update_rank_indices()

    def _update_rank_indices(self) -> None:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        order = torch.randperm(len(self.video_indices), generator=generator).tolist()

        # Greedy assignment of videos to replicas to balance frame counts while
        # keeping complete videos on each replica.
        per_replica_videos: List[List[int]] = [[] for _ in range(self.num_replicas)]
        per_replica_lengths = [0 for _ in range(self.num_replicas)]
        for vid_idx in order:
            target = min(range(self.num_replicas), key=per_replica_lengths.__getitem__)
            per_replica_videos[target].append(vid_idx)
            per_replica_lengths[target] += len(self.video_indices[vid_idx])

        self._rank_indices = []
        max_length = 0
        for videos in per_replica_videos:
            indices: List[int] = []
            for vid_idx in videos:
                indices.extend(self.video_indices[vid_idx])
            if not indices:
                raise ValueError('Each replica must receive at least one video '
                                 'when using DistributedSequentialVideoSampler.')
            self._rank_indices.append(indices)
            max_length = max(max_length, len(indices))

        # Pad with the last frame index so every replica yields the same number
        # of samples, matching DistributedSampler semantics.
        for indices in self._rank_indices:
            if len(indices) < max_length:
                pad_value = indices[-1]
                indices.extend([pad_value] * (max_length - len(indices)))

        self.num_samples = max_length
        self.total_size = max_length * self.num_replicas

    def __iter__(self) -> Iterable[int]:
        return iter(self._rank_indices[self.rank])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._update_rank_indices()
