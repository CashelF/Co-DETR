from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadSeqInfo:
    """Load sequence info from img_info to results dict."""
    
    def __call__(self, results):
        info = results.get('img_info', {})
        for key in ['video_id', 'frame_id', 'is_video_first']:
            if key in info:
                results[key] = info[key]
        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'
