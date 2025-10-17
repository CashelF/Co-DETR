from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    """Freeze backbone params; optional eval() to freeze BN stats."""
    def __init__(self, module_names=('backbone',), set_eval=True):
        self.module_names = module_names
        self.set_eval = set_eval

    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        for name in self.module_names:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            # stop grads
            for p in mod.parameters():
                p.requires_grad = False
            # freeze BN stats etc. (safe for ResNet; ViT usually has LN/GN)
            if self.set_eval:
                mod.eval()
