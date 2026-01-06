
from mmcv.runner import HOOKS, Hook
from mmdet.datasets import build_dataloader, build_dataset
import torch
from mmcv.parallel import collate, scatter
import mmcv

@HOOKS.register_module()
class ValLossHook(Hook):
    def __init__(self, val_dataset_cfg, interval=1, samples_per_gpu=1, workers_per_gpu=2, **kwargs):
        self.val_dataset_cfg = val_dataset_cfg
        self.interval = interval
        self.samples_per_gpu = samples_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.dataloader = None

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
            
        runner.logger.info(f'Epoch [{runner.epoch + 1}] - Running validation loss calculation...')
        
        if self.dataloader is None:
            dataset = build_dataset(self.val_dataset_cfg)
            self.dataloader = build_dataloader(
                dataset,
                samples_per_gpu=self.samples_per_gpu,
                workers_per_gpu=self.workers_per_gpu,
                dist=runner.world_size > 1,
                shuffle=False
            )
            
        model = runner.model
        model.eval()
        
        losses = {}
        batch_count = 0
        
        prog_bar = mmcv.ProgressBar(len(self.dataloader))
        
        for i, data in enumerate(self.dataloader):
            with torch.no_grad():
                # Simple scatter
                if torch.cuda.is_available():
                    device = next(model.parameters()).device
                    if runner.world_size == 1:
                         data = scatter(data, [device.index])[0]
                    else:
                        # In DDP, the model is WrapDistributed.
                        # data is from distributed sampler.
                        # scatter handles it.
                        pass
                
                if runner.world_size > 1:
                     # DDP
                     # scatter requires list of input
                     data = scatter(data, [torch.cuda.current_device()])[0]
                else:
                     data = scatter(data, [next(model.parameters()).device.index])[0]
                
                
                # Forward train
                # If DDP, model(..., return_loss=True) calls forward which calls forward_train.
                loss_dict = model(**data, return_loss=True)
                
                # loss_dict contains tensors.
                for name, val in loss_dict.items():
                    if 'loss' in name:
                        if isinstance(val, torch.Tensor):
                           val = val.item()
                        elif isinstance(val, list):
                           # some heads return list of tensors
                           val = sum(x.item() for x in val)
                        
                        if name not in losses:
                            losses[name] = 0.0
                        losses[name] += val
                        
            batch_count += 1
            prog_bar.update()
            
        # Average and log
        log_dict = {}
        for name, total_loss in losses.items():
            avg_loss = total_loss / batch_count
            log_dict[f'val_{name}'] = avg_loss
            
        # Add total val loss
        total_val_loss = sum(log_dict.values())
        log_dict['val_loss'] = total_val_loss
        
        runner.log_buffer.update(log_dict)
        runner.logger.info(f'Val Loss: {total_val_loss:.4f}')

        # Robust Fallback: Explicitly log to WandB if active
        import wandb
        if wandb.run is not None:
             wandb.log(log_dict, commit=False) # commit=False so we don't increment step prematurely? 
                                               # LoggerHook usually commits.


