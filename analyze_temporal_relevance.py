
import torch
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import types
import projects
import mmcv
import os

def analyze_relevance(config_path, checkpoint_path, current_img_path, prev_img_path, wrong_img_path, device='cuda:0'):
    cfg = Config.fromfile(config_path)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    model = model.to(device)
    model.eval()

    # Capture Weights Logic
    captured_weights = {} # {layer_idx: tensor}
    
    def capture_forward(layer_idx, original_forward):
        def forward(self, *args, **kwargs):
            kwargs['need_weights'] = True
            outputs = original_forward(*args, **kwargs)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                w = outputs[1]
                if w is not None:
                    captured_weights[layer_idx] = w.detach().cpu()
            return outputs
        return forward

    # Patch Decoder
    decoder = model.query_head.transformer.decoder
    for i, layer in enumerate(decoder.layers):
        self_attn = layer.attentions[0]
        target_module = self_attn
        if hasattr(self_attn, 'attn') and isinstance(self_attn.attn, torch.nn.MultiheadAttention):
            target_module = self_attn.attn
        orig_forward = target_module.forward
        target_module.forward = types.MethodType(capture_forward(i, orig_forward), target_module)

    # Helper to prepare image tensor and meta
    def prepare_data(img_path_or_array, filename_override):
        if isinstance(img_path_or_array, str):
            img_real = mmcv.imread(img_path_or_array)
        else:
            img_real = img_path_or_array # assume array
            
        H, W = 800, 1000
        img_real = mmcv.imresize(img_real, (W, H))
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        img_norm = mmcv.imnormalize(img_real, mean, std, to_rgb=True)
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        meta = [{
            'img_shape': (H, W, 3),
            'ori_shape': (H, W, 3),
            'pad_shape': (H, W, 3),
            'batch_input_shape': (H, W),
            'scale_factor': 1.0,
            'ori_filename': filename_override,
            'flip': False,
            'flip_direction': None,
            'img_norm_cfg': {'mean': mean, 'std': std, 'to_rgb': True}
        }]
        return img_tensor, meta

    def run_experiment(name, prev_source, current_source):
        print(f"\n--- Running Experiment: {name} ---")
        captured_weights.clear()
        
        # 1. Run Previous Frame (to populate cache)
        # We give it filename ending in _000
        p_tensor, p_meta = prepare_data(prev_source, 'simulated_video_000.jpg')
        with torch.no_grad():
            x = model.extract_feat(p_tensor, p_meta)
            model.query_head.forward(x, p_meta)
            
        # 2. Run Current Frame (reading cache)
        # We give it filename ending in _001 so it matches sequence logic
        c_tensor, c_meta = prepare_data(current_source, 'simulated_video_001.jpg')
        captured_weights.clear() # Clear weights from prev run
        
        with torch.no_grad():
            x = model.extract_feat(c_tensor, c_meta)
            model.query_head.forward(x, c_meta)
            
        # Analyze average attention to valid prev queries (frames)
        # We aggregate across all layers (or just print per layer)
        results = {}
        for i, w in captured_weights.items():
            mat = w[0] # Head avg
            total_q = mat.shape[0]
            num_query = cfg.model.query_head.num_query
            
            if total_q < num_query * 1.5: continue
            
            content_rows = mat[:num_query, :]
            attn_to_prev = content_rows[:, num_query:].sum(dim=1).mean().item()
            results[i] = attn_to_prev
            
        return results

    # Data
    # True Previous
    # We don't have the explicit 'prev' path from user, but we can assume sequential naming if user provided one.
    # User provided: .../1A2_1A2_0008.jpg
    # We can try .../1A2_1A2_0007.jpg as prev.
    
    # We need to construct paths.
    # Check if files exist.
    
    base_dir = os.path.dirname(current_img_path)
    cur_fname = os.path.basename(current_img_path)
    # Parse 1A2_1A2_0008.jpg -> 0008 -> 0007
    # Assuming format ending in _XXXX.jpg
    try:
        prefix, num_str = cur_fname.rsplit('_', 1)
        num_str, ext = num_str.split('.')
        curr_num = int(num_str)
        prev_num = curr_num - 1
        prev_fname = f"{prefix}_{prev_num:04d}.{ext}"
        derived_prev_path = os.path.join(base_dir, prev_fname)
        
        if not os.path.exists(derived_prev_path):
            print(f"Warning: Derived prev path {derived_prev_path} does not exist.")
            derived_prev_path = prev_img_path # Fallback or fail
    except:
        derived_prev_path = prev_img_path

    # Random Noise Image
    noise_img = np.random.randint(0, 255, (800, 1000, 3), dtype=np.uint8)
    
    # Run
    res_true = run_experiment("True Previous", derived_prev_path, current_img_path)
    res_wrong = run_experiment("Wrong Previous", wrong_img_path, current_img_path)
    res_noise = run_experiment("Random Noise", noise_img, current_img_path)
    
    print("\n\n=== Comparative Results (Attention to Previous Queries) ===")
    print(f"{'Layer':<10} | {'True':<12} | {'Wrong':<12} | {'Noise':<12}")
    print("-" * 55)
    
    layers = sorted(res_true.keys())
    for l in layers:
        v_true = res_true.get(l, 0)
        v_wrong = res_wrong.get(l, 0)
        v_noise = res_noise.get(l, 0)
        print(f"{l:<10} | {v_true:.4f}       | {v_wrong:.4f}       | {v_noise:.4f}")
        
    print("\nInterpretation:")
    print("Higher values mean the model is paying MORE attention to the temporal context.")
    # Usually: True > Wrong > Noise.
    # Or True ~= Wrong (if semantic overlap is high or if model just likes 'any' history) > Noise.

if __name__ == '__main__':
    # User provided current: /data/cashel-data/abes-gladius-data-vid/train/images/1A2_1A2_0008.jpg
    # We need a 'wrong' image. We can just pick another one from the same dir if we could list it, 
    # or just use ..._0000.jpg if it exists?
    # Or just use the same image as 'wrong' (self-history)? No, that would be high match.
    # Let's use a dummy path for 'prev' and 'wrong' and let the derivation logic work or fail?
    # I'll try to find a different image for 'wrong'.
    
    current = '/data/cashel-data/abes-gladius-data-vid/train/images/1A2_1A2_0008.jpg'
    # Wrong: 1A2_1A2_0001.jpg (distant frame)
    wrong = '/data/cashel-data/abes-gladius-data-vid/train/images/1A2_1A2_0001.jpg' 
    # Prev (fallback): 1A2_1A2_0007.jpg (User didn't provide, but I derive it)
    
    analyze_relevance(
        'projects/configs/co_dino_vit/gladius_vit_large_co_dino.py',
        'work_dir/cashel_test_2bs/latest.pth',
        current,
        'DUMMY_PATH_WILL_BE_DERIVED', 
        wrong
    )
