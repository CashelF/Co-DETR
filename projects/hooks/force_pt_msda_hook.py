from mmcv.runner import Hook, HOOKS

@HOOKS.register_module()
class ForcePTMSDAHook(Hook):
    """Force MS-Deformable-Attention CUDA ops to run in FP32 under AMP.

    Patches ONLY low-level ops:
      - mmcv.ops.multi_scale_deform_attn.multi_scale_deform_attn (function)
      - mmcv.ops.multi_scale_deform_attn.MultiScaleDeformableAttnFunction.apply (autograd)
    """

    _patched = False

    def before_run(self, runner):
        if ForcePTMSDAHook._patched:
            return
        logger = getattr(runner, 'logger', None)

        try:
            import torch
            import mmcv.ops.multi_scale_deform_attn as msda_mod
        except Exception as e:
            if logger:
                logger.warning(f'[ForcePTMSDAHook] Cannot import MSDA ops: {e}')
            return

        # ----------------- helper: wrap a callable to cast tensors & disable autocast -----------------
        def _wrap_msda_callable(callable_obj, name_for_log):
            def _wrapper(*args, **kwargs):
                # Positional/kw tolerant extraction of tensors we must upcast
                # Known arg names by MMCV 1.5: value, sampling_loc / sampling_locations, attn_weight / attention_weights
                def pick_arg(i, k):  # i: pos index, k: kw name
                    if k in kwargs:
                        return kwargs[k]
                    return args[i] if len(args) > i else None

                value = pick_arg(0, 'value')
                # function uses 'sampling_loc', autograd uses same; bricks use 'sampling_locations' but we don't patch bricks
                sampling = kwargs.get('sampling_loc', None)
                if sampling is None and len(args) > 3:
                    sampling = args[3]
                attn_w = kwargs.get('attn_weight', None)
                if attn_w is None and len(args) > 4:
                    attn_w = args[4]

                if hasattr(value, 'is_cuda') and value.is_cuda:
                    with torch.cuda.amp.autocast(enabled=False):
                        # upcast tensors we pass through
                        if sampling is not None:
                            if 'sampling_loc' in kwargs:
                                kwargs['sampling_loc'] = sampling.float()
                            # positional index 3 is sampling_loc for the function/Function.apply
                            elif len(args) > 3:
                                args = list(args)
                                args[3] = args[3].float()
                                args = tuple(args)

                        if attn_w is not None:
                            if 'attn_weight' in kwargs:
                                kwargs['attn_weight'] = attn_w.float()
                            elif len(args) > 4:
                                args = list(args)
                                args[4] = args[4].float()
                                args = tuple(args)

                        # value is always first positional for the ops-level calls
                        if value is not None:
                            if 'value' in kwargs:
                                kwargs['value'] = value.float()
                            elif len(args) > 0:
                                args = list(args)
                                args[0] = args[0].float()
                                args = tuple(args)

                        out = callable_obj(*args, **kwargs)
                    return out.to(value.dtype if hasattr(value, 'dtype') else out.dtype)
                else:
                    return callable_obj(*args, **kwargs)

            _wrapper.__name__ = f'{name_for_log}_fp32_wrapper'
            return _wrapper

        # -------- 1) Patch the module-level function
        try:
            orig_func = getattr(msda_mod, 'multi_scale_deform_attn', None)
            if orig_func is not None and not getattr(msda_mod, '_codetr_func_wrapped', False):
                msda_mod.multi_scale_deform_attn = _wrap_msda_callable(orig_func, 'multi_scale_deform_attn')
                msda_mod._codetr_func_wrapped = True
                if logger:
                    logger.info('[ForcePTMSDAHook] Wrapped ops function multi_scale_deform_attn -> FP32.')
        except Exception as e:
            if logger:
                logger.warning(f'[ForcePTMSDAHook] Failed wrapping function: {e}')

        # -------- 2) Patch the autograd Function.apply
        try:
            Func = getattr(msda_mod, 'MultiScaleDeformableAttnFunction', None)
            if Func is not None and not getattr(Func, '_codetr_apply_wrapped', False):
                orig_apply = Func.apply
                Func.apply = _wrap_msda_callable(orig_apply, 'MSDAFunction.apply')
                Func._codetr_apply_wrapped = True
                if logger:
                    logger.info('[ForcePTMSDAHook] Wrapped autograd MSDA Function.apply -> FP32.')
        except Exception as e:
            if logger:
                logger.warning(f'[ForcePTMSDAHook] Failed wrapping Function.apply: {e}')

        ForcePTMSDAHook._patched = True
        if logger:
            logger.info('[ForcePTMSDAHook] MSDA FP32 ops patches installed.')
