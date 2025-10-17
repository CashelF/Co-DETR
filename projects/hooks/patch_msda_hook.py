from mmcv.runner import Hook, HOOKS

@HOOKS.register_module()
class PatchMSDAHook(Hook):
    def before_run(self, runner):
        logger = getattr(runner, 'logger', None)
        try:
            import mmcv.ops.multi_scale_deform_attn as _  # noqa: F401
            if logger:
                logger.info('[PatchMSDAHook] MSDA module is available.')
        except Exception as e:
            if logger:
                logger.warning(f'[PatchMSDAHook] Failed to import MSDA: {e}')
