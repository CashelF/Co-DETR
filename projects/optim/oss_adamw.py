# projects/optim/oss_adamw.py
import torch
from fairscale.optim.oss import OSS
from mmcv.runner import OPTIMIZERS

@OPTIMIZERS.register_module()
class OSSAdamW(OSS):
    """Wrap AdamW with FairScale OSS (ZeRO-1) for optimizer-state sharding."""
    def __init__(self, params, lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, optim=torch.optim.AdamW,
                         lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
