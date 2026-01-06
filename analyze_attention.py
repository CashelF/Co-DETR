
import torch
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import types
import projects # Register custom models

def analyze_attention(config_path, checkpoint_path, image_path, device='cuda:0'):
    cfg = Config.fromfile(config_path)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        load_checkpoint(model, checkpoint_path, map_location='cpu')
    else:
        print("No checkpoint found, using random weights (analysis might be meaningless)")

    model = model.to(device)
    model.eval()

    # Storage for weights
    # List of (layer_idx, weights_tensor)
    # weights_tensor shape: (Bs, Num_Heads, Total_Q, Total_Q)
    captured_weights = {}

    def capture_forward(layer_idx, original_forward):
        def forward(self, *args, **kwargs):
            # Force outputting weights
            kwargs['need_weights'] = True
            
            # Call original
            outputs = original_forward(*args, **kwargs)
            
            # outputs is (attn_output, attn_output_weights)
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                w = outputs[1]
                if w is not None:
                    captured_weights[layer_idx] = w.detach().cpu()
            
            return outputs
        return forward

    # Patch the decoder layers
    print("Patching decoder attention layers...")
    decoder = model.query_head.transformer.decoder
    for i, layer in enumerate(decoder.layers):
        # Determine which attention is self-attn. 
        self_attn = layer.attentions[0]
        print(f"Layer {i} attn[0] type: {type(self_attn)}")
        
        target_module = self_attn
        
        # If it's MMCV wrapper, try to find the inner torch module
        if hasattr(self_attn, 'attn') and isinstance(self_attn.attn, torch.nn.MultiheadAttention):
            print(f"  Found inner torch.nn.MultiheadAttention. Patching it.")
            target_module = self_attn.attn
        else:
             print(f"  Patching the module directly (hope it returns weights with need_weights=True).")

        orig_forward = target_module.forward
        target_module.forward = types.MethodType(capture_forward(i, orig_forward), target_module)

    import mmcv
    print(f"Loading image from {image_path}")
    img_real = mmcv.imread(image_path)
    # Resize to H, W
    H, W = 800, 1000
    img_real = mmcv.imrescale(img_real, (W, H)) # imrescale takes (max_long_edge) or scale or (w, h) tuple?
    # mmcv.imrescale(img, scale) -> scale can be float or tuple. If tuple, it's (max_long, max_short).
    # Easier to use mmcv.imresize
    img_real = mmcv.imresize(img_real, (W, H)) 
    
    # Normalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)
    img_norm = mmcv.imnormalize(img_real, mean, std, to_rgb=True)
    
    # To Tensor (C, H, W)
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    img_metas = [{
        'img_shape': (H, W, 3),
        'ori_shape': (H, W, 3),
        'pad_shape': (H, W, 3),
        'batch_input_shape': (H, W),
        'scale_factor': 1.0,
        'ori_filename': 'video_frame_000.jpg', # First frame
        'flip': False,
        'flip_direction': None,
        'img_norm_cfg': {'mean': mean, 'std': std, 'to_rgb': True}
    }]
    
    # 1. First Run (Populate Cache)
    with torch.no_grad():
        x = model.extract_feat(img_tensor, img_metas)
        model.query_head.forward(x, img_metas)
    
    print("First run complete. Cache should be populated.")
    
    # 2. Second Run (Use Cache)
    # Change filename to trigger same sequence
    img_metas[0]['ori_filename'] = 'video_frame_001.jpg' 
    
    captured_weights.clear()
    
    with torch.no_grad():
        x = model.extract_feat(img_tensor, img_metas)
        model.query_head.forward(x, img_metas)
        
    print(f"Second run complete. Captured weights for {len(captured_weights)} layers.")
    
    # Analyze
    if not captured_weights:
        print("No weights captured! Did the model run the decoder?")
        return
        
    # Stats
    # We need to identify appropriate ranges.
    # DINO: [DN Queries (fixed/dynamic?) | Content Queries (N=1500?) | Prev Queries (N=1500?)]
    # DN queries are only present during training usually? 
    # In `simple_test` / `forward_dummy`, DN might be disabled?
    # Let's check `CoDINOHead.forward`.
    # It calls `dn_generator`?
    # validation/test usually doesn't use DN.
    # So likely just [Content | Prev].
    
    # Transformer.forward:
    # base_query_num = query_embed.shape[1] (Content)
    # prev_query_num = ...
    
    # If DN is off:
    # Rows 0..Content are Content.
    # Rows Content..End are Prev.
    
    # Columns 0..Content are Content.
    # Columns Content..End are Prev.
    
    for i, w in captured_weights.items():
        # w shape: (Batch, Total_Q, Total_Q) (averaged heads)
        # Assuming Batch=1
        mat = w[0] # (Total, Total)
        
        total_q = mat.shape[0]
        # We need to know Content Num vs Prev Num.
        # We can guess: total_q = 3000 (1500 + 1500)?
        # Or check config. num_queries=1500 usually for Co-DINO-L.
        
        # Let's assume split is roughly half-half if cache hit.
        # Or look for the block boundary.
        
        half = total_q // 2
        # Use a heuristic or exact config if possible.
        # cfg.model.query_head.num_query
        num_query = cfg.model.query_head.num_query
        
        # If total_q approx 2 * num_query, then we have prev queries.
        if total_q < num_query * 1.5:
             print(f"Layer {i}: Total Q {total_q}. Seems NO prev queries used? (Exp {2*num_query})")
             continue
             
        # Content Queries: Indices [0 : num_query] (Assuming no DN in eval)
        # Prev Queries: Indices [num_query : ]
        
        # Attention from Content to Prev
        # Rows: 0..num_query
        # Cols: num_query..end
        
        content_rows = mat[:num_query, :]
        
        attn_to_self = content_rows[:, :num_query].sum(dim=1).mean().item()
        attn_to_prev = content_rows[:, num_query:].sum(dim=1).mean().item()
        
        print(f"Layer {i}:")
        print(f"  Avg Attn to Content (Self/Learned): {attn_to_self:.4f}")
        print(f"  Avg Attn to Previous Frame:         {attn_to_prev:.4f}")
        print(f"  Ratio (Prev / Total):               {attn_to_prev / (attn_to_self + attn_to_prev):.2%}")

import sys
import os
if __name__ == '__main__':
    analyze_attention(
        'projects/configs/co_dino_vit/gladius_vit_large_co_dino.py', 
        'work_dir/cashel_test_2bs/latest.pth',
        '/data/cashel-data/abes-gladius-data-vid/train/images/1A2_1A2_0008.jpg'
    )
