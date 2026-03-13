_base_ = './maritime_0128_nodes.py'

data_root = '/data/cashel-data/marvel-maritime-images'

data = dict(
    test=dict(
        ann_file=data_root + '/dummy_labels.json',
        img_prefix=data_root + '/images/',
    )
)

# We want to use format-only so evaluate metric isn't complaining about dummy bboxes
evaluation = dict(interval=1000, metric='bbox')
