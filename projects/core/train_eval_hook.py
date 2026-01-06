
from mmcv.runner import HOOKS, Hook
from mmdet.datasets import build_dataloader, build_dataset
import torch
import mmcv
from mmdet.apis import single_gpu_test
import numpy as np
from collections import defaultdict
import math
import copy

@HOOKS.register_module()
class TrainEvalHook(Hook):
    def __init__(self, dataset_cfg, interval=1, samples_per_gpu=1, workers_per_gpu=2, subset_ratio=1.0, **kwargs):
        self.dataset_cfg = dataset_cfg
        self.interval = interval
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.subset_ratio = subset_ratio
        self.dataloader = None
        self.eval_kwargs = kwargs

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
            
        runner.logger.info(f'Epoch [{runner.epoch + 1}] - Preparing training evaluation on {self.subset_ratio:.0%} subset...')
        
        if self.dataloader is None:
            # Build dataset logic
            dataset = build_dataset(self.dataset_cfg)
            
            # Subsetting Logic
            if self.subset_ratio < 1.0:
                indices_to_use = self._get_sequential_subset_indices(dataset, self.subset_ratio)
                
                # Destructive slicing to maintain compatibility with .evaluate()
                # We assume CocoVideoDataset structure (data_infos, img_ids)
                if hasattr(dataset, 'data_infos'):
                    dataset.data_infos = [dataset.data_infos[i] for i in indices_to_use]
                    if hasattr(dataset, 'img_ids'):
                         dataset.img_ids = [dataset.img_ids[i] for i in indices_to_use]
                         
                    # MMDetection 2.x datasets might have other attributes like 'proposals'
                    if hasattr(dataset, 'proposals') and dataset.proposals is not None:
                         dataset.proposals = [dataset.proposals[i] for i in indices_to_use]
                         
                    runner.logger.info(f'Sequential Subset Created: {len(dataset)} samples. (Videos preserved)')
                else:
                    runner.logger.warning("Dataset does not have 'data_infos'. Subsetting might fail or be skipped.")
                    # Fallback to Subset wrapper if necessary, but risky for evaluate()
                    dataset = torch.utils.data.Subset(dataset, indices_to_use)

            self.dataloader = build_dataloader(
                dataset,
                samples_per_gpu=self.samples_per_gpu,
                workers_per_gpu=self.workers_per_gpu,
                dist=runner.world_size > 1,
                shuffle=False # Crucial for sequential integrity
            )
            
        model = runner.model
        model.eval()
        
        results = None
        if runner.world_size > 1:
            from mmdet.apis import multi_gpu_test
            results = multi_gpu_test(model, self.dataloader, gpu_collect=True)
        else:
            results = single_gpu_test(model, self.dataloader, show=False)
            
        if runner.rank == 0:
            if results is None:
                runner.logger.warning("Train evaluation returned No results.")
                return

            runner.logger.info(f'Evaluating train subset results...')
            
            # Handle potential mismatch if we used Subset wrapper (fallback path)
            eval_dataset = self.dataloader.dataset
            if isinstance(eval_dataset, torch.utils.data.Subset):
                 # This path is risky for evaluate(). Try enabling it on the underlying dataset?
                 # Usually underlying.evaluate expects full results.
                 # We skip if fallback hit.
                 runner.logger.warning("Skipping metric calculation because Subset wrapper was used (incompatible with evaluate).")
                 return

            eval_res = eval_dataset.evaluate(results, logger=runner.logger, **self.eval_kwargs)
            
            log_dict = {}
            for name, val in eval_res.items():
                log_dict[f'train_{name}'] = val
                
            runner.log_buffer.update(log_dict)
            
            # Robust Fallback: Explicitly log to WandB
            import wandb
            if wandb.run is not None:
                wandb.log(log_dict, commit=False)

            
    def _get_sequential_subset_indices(self, dataset, ratio):
        if hasattr(dataset, 'dataset'): 
             real_dataset = dataset.dataset
        else:
             real_dataset = dataset
             
        if not hasattr(real_dataset, 'data_infos'):
            return list(range(len(dataset))) # No subsetting possible

        video_to_indices = defaultdict(list)
        for idx, info in enumerate(real_dataset.data_infos):
            # Try to find video ID
            vid = info.get('video_id', None)
            if vid is None:
                # Try 'seq_id'
                vid = info.get('seq_id', None)
            if vid is None:
                # Fallback to parsing filename if available
                fname = info.get('ori_filename', '')
                if '_' in fname:
                     vid = fname.rsplit('_', 1)[0]
                else:
                     vid = fname # worst case
            
            video_to_indices[vid].append(idx)
            
        all_videos = list(video_to_indices.keys())
        all_videos.sort()
        
        # Deterministic shuffle
        rng = np.random.RandomState(42)
        rng.shuffle(all_videos)
        
        num_keep = max(1, int(math.ceil(len(all_videos) * ratio)))
        keep_videos = set(all_videos[:num_keep])
        
        final_indices = []
        for vid in keep_videos:
            final_indices.extend(video_to_indices[vid])
            
        final_indices.sort()
        return final_indices
