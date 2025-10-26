"""COCO-style video dataset for sequential sampling."""

from __future__ import annotations

from collections import defaultdict

from .api_wrappers import COCO
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class CocoVideoDataset(CocoDataset):
    """COCO detection dataset that keeps frames grouped by video.

    This minimal implementation extends :class:`CocoDataset` with sequence
    awareness so that dataloaders can iterate through frames in temporal order
    while still leveraging the standard COCO evaluation utilities. When
    ``load_as_video`` is ``True`` (the default) the annotation file is expected
    to include top-level ``videos`` entries as well as ``video_id`` and
    ``frame_id`` fields for every image. Frames are sorted by ``frame_id``
    within each video and emitted sequentially.
    """

    def __init__(self, *args, load_as_video: bool = True, **kwargs):
        self.load_as_video = load_as_video
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        if not self.load_as_video:
            return super().load_annotations(ann_file)

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        videos = self.coco.dataset.get('videos')
        if not videos:
            raise KeyError(
                'COCO video annotations must contain a top-level "videos" list ')

        video_meta = {video['id']: video for video in videos if 'id' in video}
        if len(video_meta) != len(videos):
            raise KeyError('Each video entry requires a unique integer "id".')

        image_groups: defaultdict[int, list] = defaultdict(list)
        for img in self.coco.dataset.get('images', []):
            if 'video_id' not in img:
                raise KeyError('Each image must contain a "video_id" field when '
                               'load_as_video=True.')
            if 'frame_id' not in img:
                raise KeyError('Each image must contain a "frame_id" field when '
                               'load_as_video=True.')
            image_groups[img['video_id']].append(img)

        data_infos = []
        self.img_ids = []
        self.vid_ids = []

        for vid_id in sorted(video_meta.keys()):
            frames = image_groups.get(vid_id, [])
            if not frames:
                continue
            frames.sort(key=lambda x: x['frame_id'])
            self.vid_ids.append(vid_id)

            for frame in frames:
                info = frame.copy()
                info['filename'] = info['file_name']
                if 'video_name' not in info:
                    info['video_name'] = video_meta[vid_id].get('name', str(vid_id))
                if 'is_video_first' not in info:
                    info['is_video_first'] = int(frame['frame_id']) == 0
                data_infos.append(info)
                self.img_ids.append(frame['id'])

        if not data_infos:
            raise ValueError('No frames were loaded from the annotation file.')

        return data_infos
