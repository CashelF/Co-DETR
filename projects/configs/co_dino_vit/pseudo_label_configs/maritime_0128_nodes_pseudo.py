_base_ = './maritime_0128_nodes.py'

# 1. Modify the test dataset to point to the training dataset
# We use the training images and the training annotations but processed
# through the test_pipeline (no multi-scale augmentations, no random flips, etc)
data = dict(
    test=dict(
        ann_file='/data/cashel-data/maritime-0128/train/labels.json',
        img_prefix='/data/cashel-data/maritime-0128/train/',
        # The base config already defines test_pipeline
        # We'll just inherit that pipeline for deterministic bbox generation
    )
)

# 2. Modify the evaluation config so it doesn't crash trying to evaluate 
# predictions vs ground truth if we are just formatting outputs
evaluation = dict(interval=1000, metric='bbox')
