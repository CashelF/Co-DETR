import math
import warnings
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE, TRANSFORMER_LAYER, ATTENTION
from mmcv.cnn.bricks.transformer import TransformerLayerSequence, MultiheadAttention

from mmdet.models.utils.transformer import Transformer, DeformableDetrTransformer, DeformableDetrTransformerDecoder, DetrTransformerDecoderLayer
from mmdet.models.utils.builder import TRANSFORMER


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CoDeformableDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, look_forward_twice=False, **kwargs):

        super(CoDeformableDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.look_forward_twice = look_forward_twice

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(
                    new_reference_points
                    if self.look_forward_twice
                    else reference_points
                )
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@TRANSFORMER.register_module()
class CoDeformableDetrTransformer(DeformableDetrTransformer):
    """Implements the DeformableDETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 mixed_selection=True,
                 with_pos_coord=True,
                 with_coord_feat=True,
                 num_co_heads=1,
                 **kwargs):
        self.mixed_selection = mixed_selection
        self.with_pos_coord = with_pos_coord
        self.with_coord_feat = with_coord_feat
        self.num_co_heads = num_co_heads
        super(CoDeformableDetrTransformer, self).__init__(**kwargs)
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                # bug: this code should be 'self.head_pos_embed = nn.Embedding(self.num_co_heads, self.embed_dims)', we keep this bug for reproducing our results with ResNet-50.
                # You can fix this bug when reproducing results with swin transformer.
                self.head_pos_embed = nn.Embedding(self.num_co_heads, 1, 1, self.embed_dims)
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims*2, self.embed_dims*2))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims*2))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        num_pos_feats = self.embed_dims // 2
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()),
                          dim=4).flatten(2)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                reg_branches=None,
                cls_branches=None,
                return_encoder_output=False,
                attn_masks=None,
                prev_query_feats_list=None,
                prev_reference_points_list=None,
                prev_query_valid_lens=None,
                return_decoder_cache=False,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when
                `with_box_refine` is True. Default to None.
            cls_branches (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage`
                 is True. Default to None.


        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        assert self.as_two_stage or query_embed is not None

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = \
            self.get_reference_points(spatial_shapes,
                                      valid_ratios,
                                      device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)

        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = \
                self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes)
            enc_outputs_class = cls_branches[self.decoder.num_layers](
                output_memory)
            enc_outputs_coord_unact = \
                reg_branches[
                    self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk = query_embed.shape[0]
            topk_proposals = torch.topk(
                enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(
                enc_outputs_coord_unact, 1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if not self.mixed_selection:
                query_pos, query = torch.split(pos_trans_out, c, dim=2)
            else:
                # query_embed here is the content embed for deformable DETR
                query = query_embed.unsqueeze(0).expand(bs, -1, -1)
                query_pos, _ = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        prev_query_feats, prev_reference_points, prev_query_valid_lens = (
            self._prepare_prev_queries(
                prev_query_feats_list,
                prev_reference_points_list,
                prev_query_valid_lens,
                bs,
                device=feat_flatten.device,
                dtype=feat_flatten.dtype))
        base_query_num = query.size(1)
        dn_query_num = 0
        detection_query_num = base_query_num
        if prev_query_feats is not None and prev_reference_points is not None:
            query, query_pos, reference_points, attn_masks = self._append_prev_queries(
                query,
                query_pos,
                reference_points,
                prev_query_feats,
                prev_reference_points,
                prev_query_valid_lens,
                attn_masks=attn_masks)

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references
        decoder_cache = None
        if return_decoder_cache:
            # If tracking (prev queries exist), we want to cache everything (New + Tracks)
            # detection_query_num currently equals base_query_num (New).
            # If we appended tracks, query size increased.
            # We should cache all detection queries (New + Track).
            # dn_query_num is 0 usually here (handled elsewhere?).
            # Valid query range is [:total_query_num] if no DN, or handled by build_decoder_cache logic.
            # _build_decoder_cache uses detection_query_num as "length".
            
            queries_to_cache = detection_query_num
            if prev_query_feats is not None:
                # If we added tracks, we want to cache them too.
                # Tracks are appended at the end.
                queries_to_cache = detection_query_num + prev_query_feats.size(1)
            
            decoder_cache = self._build_decoder_cache(
                inter_states,
                inter_references_out,
                queries_to_cache,
                dn_query_num)
        if self.as_two_stage:
            if return_encoder_output:
                return inter_states, init_reference_out,\
                    inter_references_out, enc_outputs_class,\
                    enc_outputs_coord_unact, memory, decoder_cache
            return inter_states, init_reference_out,\
                inter_references_out, enc_outputs_class,\
                enc_outputs_coord_unact, decoder_cache
        if return_encoder_output:
            return inter_states, init_reference_out, \
                inter_references_out, None, None, memory, decoder_cache
        return inter_states, init_reference_out, \
            inter_references_out, None, None, decoder_cache

    def _prepare_prev_queries(self,
                              prev_query_feats_list,
                              prev_reference_points_list,
                              prev_query_valid_lens,
                              batch_size,
                              device,
                              dtype):
        if prev_query_feats_list is None or prev_reference_points_list is None:
            return None, None, None
        if not isinstance(prev_query_feats_list, (list, tuple)) or not isinstance(prev_reference_points_list, (list, tuple)):
            raise ValueError('prev_query_feats_list and prev_reference_points_list must be lists when provided.')
        if len(prev_query_feats_list) != batch_size or len(prev_reference_points_list) != batch_size:
            raise ValueError('prev query lists must have length equal to batch size.')

        if prev_query_valid_lens is None:
            prev_query_valid_lens = [None for _ in range(batch_size)]
        elif torch.is_tensor(prev_query_valid_lens):
            prev_query_valid_lens = prev_query_valid_lens.tolist()
        elif not isinstance(prev_query_valid_lens, (list, tuple)):
            prev_query_valid_lens = [prev_query_valid_lens for _ in range(batch_size)]

        ref_dim = None
        max_prev = 0
        valid_lengths = []
        normalized_feats = []
        normalized_refs = []
        def _normalize_length(value, default_len):
            if value is None:
                return default_len
            while isinstance(value, (list, tuple)):
                if not value:
                    return 0
                value = value[0]
            try:
                return int(value)
            except (TypeError, ValueError):
                return default_len
        for feats, refs, length in zip(prev_query_feats_list, prev_reference_points_list, prev_query_valid_lens):
            if feats is None or refs is None:
                normalized_feats.append(None)
                normalized_refs.append(None)
                valid_lengths.append(0)
                continue
            feats = torch.as_tensor(feats, device=device, dtype=dtype)
            refs = torch.as_tensor(refs, device=device, dtype=dtype)
            if feats.dim() != 2 or feats.size(-1) != self.embed_dims:
                raise ValueError('prev_query_feats must have shape [num, embed_dims].')
            if refs.dim() != 2:
                raise ValueError('prev_reference_points must have shape [num, ref_dim].')
            if ref_dim is None:
                ref_dim = refs.size(-1)
            valid_len = _normalize_length(length, feats.size(0))
            valid_len = min(valid_len, feats.size(0), refs.size(0)) if feats.numel() > 0 else 0
            max_prev = max(max_prev, valid_len)
            normalized_feats.append(feats)
            normalized_refs.append(refs)
            valid_lengths.append(valid_len)

        if max_prev <= 0:
            return None, None, valid_lengths

        if ref_dim is None:
            ref_dim = 4
        prev_query_feats = torch.zeros((batch_size, max_prev, self.embed_dims), device=device, dtype=dtype)
        prev_reference_points = torch.zeros((batch_size, max_prev, ref_dim), device=device, dtype=dtype)
        for idx in range(batch_size):
            valid_len = valid_lengths[idx]
            if valid_len <= 0:
                continue
            prev_query_feats[idx, :valid_len] = normalized_feats[idx][:valid_len]
            prev_reference_points[idx, :valid_len] = normalized_refs[idx][:valid_len]
        return prev_query_feats, prev_reference_points, valid_lengths

    def _append_prev_queries(self,
                             query,
                             query_pos,
                             reference_points,
                             prev_query_feats,
                             prev_reference_points,
                             prev_query_valid_lens,
                             attn_masks=None):
        bs, base_query_num, c = query.shape
        if query_pos is None:
            query_pos = query.new_zeros(bs, base_query_num, c)
        max_prev = prev_query_feats.size(1)
        total_query_num = base_query_num + max_prev
        new_query = query.new_zeros(bs, total_query_num, c)
        new_query[:, :base_query_num] = query
        new_query_pos = query_pos.new_zeros(bs, total_query_num, c)
        new_query_pos[:, :base_query_num] = query_pos
        ref_dim = reference_points.size(-1)
        new_reference = reference_points.new_zeros(bs, total_query_num, ref_dim)
        new_reference[:, :base_query_num] = reference_points
        new_reference[:, :base_query_num] = reference_points

        # Vectorized processing of previous queries
        # 1. Inverse Sigmoid
        prev_unact = inverse_sigmoid(prev_reference_points)
        
        # 2. Positional Embedding
        prev_pos_embed = self.get_proposal_pos_embed(prev_unact)
        
        # 3. Projection
        prev_pos_trans = self.pos_trans(prev_pos_embed)
        prev_pos_trans = self.pos_trans_norm(prev_pos_trans)
        
        # 4. Split
        prev_pos_all, _ = torch.split(prev_pos_trans, c, dim=2)

        for b in range(bs):
            valid_len = min(int(prev_query_valid_lens[b]), max_prev)
            if valid_len <= 0:
                continue
            new_query[b, base_query_num:base_query_num + valid_len] = prev_query_feats[b, :valid_len]
            
            # Use pre-computed positions
            new_query_pos[b, base_query_num:base_query_num + valid_len] = prev_pos_all[b, :valid_len]
            new_reference[b, base_query_num:base_query_num + valid_len] = prev_reference_points[b, :valid_len]
        if attn_masks is not None:
            if attn_masks.dim() == 2:
                attn_masks = attn_masks.unsqueeze(0).repeat(bs, 1, 1)

            if attn_masks.dim() != 3:
                raise ValueError('attn_masks is expected to have 3 dimensions when using prev queries.')
            orig_q = attn_masks.size(-1)
            # expand attn_masks
            new_attn_masks = attn_masks.new_zeros(bs, total_query_num, total_query_num).bool() # Ensure boolean
            # Usually strict masking is safer (True=masked). Co-DETR/DN-DETR uses boolean masks mostly.
            
            new_attn_masks[:, :orig_q, :orig_q] = attn_masks
            
            for b_idx in range(bs):
                valid_l = min(int(prev_query_valid_lens[b_idx]), max_prev)
                
                # Mask out invalid (padding) previous queries
                if valid_l < max_prev:
                    # Invalid prev queries cannot see anything (or be seen)
                    # BUT if we mask EVERYTHING, Softmax calculation becomes NaN (all -inf).
                    # So we MUST allow them to see THEMSELVES (diagonal).
                    
                    # 1. Mask columns (others can't see them) -> Safe
                    new_attn_masks[b_idx, :, base_query_num + valid_l:] = True
                    
                    # 2. Mask rows (they can't see others) -> Safe
                    new_attn_masks[b_idx, base_query_num + valid_l:, :] = True
                    
                    # 3. Unmask diagonal for these invalid queries so they have at least one valid target (themselves)
                    # This prevents NaN in Softmax.
                    invalid_indices = torch.arange(base_query_num + valid_l, total_query_num, device=new_attn_masks.device)
                    new_attn_masks[b_idx, invalid_indices, invalid_indices] = False
                
                # Copy DN masking pattern for Valid Previous Queries
                # We assume the last "normal" query (last col of orig mask) represents the desired visibility
                # for normal detection queries (which previous queries are).
                # This ensures previous queries cannot see DN queries (if normal queries can't).
                if base_query_num > 0:
                    example_row_mask = attn_masks[b_idx, -1, :base_query_num]
                    # Broadcast to new rows
                    new_attn_masks[b_idx, base_query_num : base_query_num + valid_l, :base_query_num] = example_row_mask

            # PyTorch MultiheadAttention expects (bs * num_heads, L, S) for 3D masks.
            # We need to repeat the mask for each head.
            # Try to get num_heads from decoder
            num_heads = 8 # Default fallback
            if hasattr(self, 'decoder') and hasattr(self.decoder, 'layers') and len(self.decoder.layers) > 0:
                 # Check attentions list
                 if hasattr(self.decoder.layers[0], 'attentions'):
                     for attn in self.decoder.layers[0].attentions:
                         if hasattr(attn, 'num_heads'):
                             num_heads = attn.num_heads
                             break
            
            new_attn_masks = new_attn_masks.repeat_interleave(num_heads, dim=0)

            return new_query, new_query_pos, new_reference, new_attn_masks
        return new_query, new_query_pos, new_reference, attn_masks

    def _build_decoder_cache(self,
                             inter_states,
                             inter_references,
                             detection_query_num,
                             dn_query_num):

        if inter_states is None or inter_references is None:

            return None
        last_hs = inter_states[-1].transpose(0, 1)
        last_refs = inter_references[-1]
        if last_refs.dim() == 4:
            last_refs = last_refs[-1]
        detection_start = dn_query_num if dn_query_num is not None else 0
        detection_length = detection_query_num if detection_query_num is not None else last_hs.size(1) - detection_start
        detection_length = max(0, min(detection_length, last_hs.size(1) - detection_start))



        cache_feats = last_hs[:, detection_start:detection_start + detection_length]
        cache_refs = last_refs[:, detection_start:detection_start + detection_length]
        

        
        cache_length = cache_feats.size(1)
        return {
            'query_feats': cache_feats.detach(),
            'reference_points': cache_refs.detach(),
            'valid_lengths': last_hs.new_full((last_hs.size(0),), cache_length, dtype=torch.long)
        }

    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        
        memory = feat_flatten
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = pos_anchors
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))
                query_pos = query_pos + self.head_pos_embed.weight[head_idx]

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, \
            inter_references_out


def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
    # TODO: It can be implemented by add an out_channel arg of
    #  mmcv.cnn.bricks.transformer.FFN
    assert num_layers > 1, \
        f'num_layers should be greater than 1 but got {num_layers}'
    h = [hidden_dim] * (num_layers - 1)
    layers = list()
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend((nn.Linear(n, k), nn.ReLU()))
    # Note that the relu func of MLP in original DETR repo is set
    # 'inplace=False', however the ReLU cfg of FFN in mmdet is set
    # 'inplace=True' by default.
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DinoTransformerDecoder(DeformableDetrTransformerDecoder):

    def __init__(self, *args, **kwargs):
        super(DinoTransformerDecoder, self).__init__(*args, **kwargs)
        self._init_layers()

    def _init_layers(self):
        self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims,
                                        self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor, pos_feat):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(
            pos_feat, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000**(2 * (dim_t // 2) / pos_feat)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()),
                            dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack(
                (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()),
                dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack(
                (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()),
                dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError('Unknown pos_tensor shape(-1):{}'.format(
                pos_tensor.size(-1)))
        return pos

    def forward(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                query_temporal_embed=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = self.gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.embed_dims//2)
            query_pos = self.ref_point_head(query_sine_embed)

            query_pos = query_pos.permute(1, 0, 2)

            if query_temporal_embed is not None:
                query_pos = query_pos + query_temporal_embed
                if lid == 0:
                    output = output + query_temporal_embed

            output = layer(
                output,
                *args,
                query_pos=query_pos,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)


            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 4
                # TODO: should do earlier
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        # Collect attention weights from TrackingDetrTransformerDecoderLayer
        previous_query_attn_proportions = []
        for layer in self.layers:
            if hasattr(layer, 'attentions') and len(layer.attentions) > 0:
                self_attn = layer.attentions[0] # Assuming first one is self-attn
                if hasattr(self_attn, 'last_attn_weights') and self_attn.last_attn_weights is not None:
                    weights = self_attn.last_attn_weights # (bs, N, N)
                    previous_query_attn_proportions.append(weights)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), previous_query_attn_proportions

        return output, reference_points, previous_query_attn_proportions

@TRANSFORMER.register_module()
class CoDinoTransformer(CoDeformableDetrTransformer):

    def __init__(self, *args, **kwargs):
        super(CoDinoTransformer, self).__init__(*args, **kwargs)

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals,
                                        self.embed_dims)
    
    def _init_layers(self):
        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims*2, self.embed_dims))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

        self.pos_trans = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
        self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        self.temporal_pos_embed = nn.Embedding(2, self.embed_dims)

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.query_embed.weight.data)
        nn.init.normal_(self.temporal_pos_embed.weight.data)


    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                prev_query_feats_list=None,
                prev_reference_points_list=None,
                prev_query_valid_lens=None,
                return_decoder_cache=False,
                reg_branches=None,
                cls_branches=None,
                **kwargs):
        assert self.as_two_stage and query_embed is None, \
            'as_two_stage must be True for DINO'
        
        # Dummy usage of pos_trans to avoid DDP errors (unused parameters in first iteration)
        # This replaces find_unused_parameters=True which conflicts with checkpointing
        if hasattr(self, 'pos_trans'):
            dummy = mlvl_feats[0].new_zeros(1, 1, self.embed_dims * 2)
            d_out = self.pos_trans_norm(self.pos_trans(dummy))
            kwargs['dummy_loss'] = d_out.sum() * 0

        if hasattr(self, 'temporal_pos_embed'):
            dummy_ind = mlvl_feats[0].new_zeros(1, 1).long()
            t_out = self.temporal_pos_embed(dummy_ind)
            if 'dummy_loss' in kwargs:
                kwargs['dummy_loss'] = kwargs['dummy_loss'] + t_out.sum() * 0
            else:
                kwargs['dummy_loss'] = t_out.sum() * 0



        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
            1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape


        prev_query_feats, prev_reference_points, prev_query_valid_lens = (
            self._prepare_prev_queries(
                prev_query_feats_list,
                prev_reference_points_list,
                prev_query_valid_lens,
                bs,
                device=feat_flatten.device,
                dtype=feat_flatten.dtype))

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](
            output_memory) + output_proposals
        cls_out_features = cls_branches[self.decoder.num_layers].out_features
        topk = self.two_stage_num_proposals
        # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk TODO
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_anchor = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embed.weight[:, None, :].repeat(1, bs,
                                                           1).transpose(0, 1)
        # NOTE the query_embed here is not spatial query as in DETR.
        # It is actually content query, which is named tgt in other
        # DETR-like models
        detection_query_num = query.size(1)
        dn_query_num = dn_label_query.size(1) if dn_label_query is not None else 0
        if dn_label_query is not None:
            query = torch.cat([dn_label_query, query], dim=1)
        if dn_bbox_query is not None:
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
        reference_points = reference_points.sigmoid()

        if prev_query_feats is not None and prev_reference_points is not None:
            query_pos = query.new_zeros(bs, query.size(1), c)
            query, query_pos, reference_points, attn_mask = self._append_prev_queries(
                query,
                query_pos,
                reference_points,
                prev_query_feats,
                prev_reference_points,
                prev_query_valid_lens,
                attn_masks=attn_mask)
        
        # Temporal Embedding
        temporal_embeddings = None
        if prev_query_feats is not None and prev_reference_points is not None:
             # 0 for current queries, 1 for previous queries
             temp_emb_indices = torch.zeros((bs, query.size(1)), dtype=torch.long, device=query.device)
             temp_emb_indices[:, detection_query_num:] = 1 
             temporal_embeddings = self.temporal_pos_embed(temp_emb_indices)
             
        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        if temporal_embeddings is not None:
            temporal_embeddings = temporal_embeddings.permute(1, 0, 2)

        inter_states, inter_references, attn_weights_list = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=attn_mask,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            query_temporal_embed=temporal_embeddings,
            **kwargs)

        inter_references_out = inter_references
        decoder_cache = None
        if return_decoder_cache:
            decoder_cache = self._build_decoder_cache(
                inter_states,
                inter_references_out,
                detection_query_num,
                dn_query_num)

        # Calculate attention metrics
        # Structure: [DN, Detection, Previous]
        # We want attention of Detection Queries -> Previous Queries
        attn_metrics = {}
        if len(attn_weights_list) > 0:
            start_det = dn_query_num
            end_det = dn_query_num + detection_query_num
            start_prev = end_det
            
            for i, weights in enumerate(attn_weights_list):
                if weights.shape[1] > start_prev: # Check if there are previous queries
                    det_to_prev_weights = weights[:, start_det:end_det, start_prev:]
                    # Sum over columns (target previous queries)
                    prev_attn_sum = det_to_prev_weights.sum(dim=-1) # (bs, num_det)
                    # Average over queries and batch
                    avg_prev_attn = prev_attn_sum.mean()
                    # Ensure metric is on the same device as query (GPU)
                    attn_metrics[f'layer_{i}_prev_attn_prop'] = avg_prev_attn.to(query.device)
                else:
                    attn_metrics[f'layer_{i}_prev_attn_prop'] = torch.tensor(0.0, device=query.device)

        return inter_states, inter_references_out, topk_score, topk_anchor, memory, decoder_cache, attn_metrics


    def forward_aux(self,
                    mlvl_feats,
                    mlvl_masks,
                    query_embed,
                    mlvl_pos_embeds,
                    pos_anchors,
                    pos_feats=None,
                    reg_branches=None,
                    cls_branches=None,
                    return_encoder_output=False,
                    attn_masks=None,
                    head_idx=0,
                    **kwargs):
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        
        memory = feat_flatten
            #enc_inter = [feat.permute(1, 2, 0) for feat in enc_inter]
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        topk = pos_anchors.shape[1]
        topk_coords_unact = inverse_sigmoid((pos_anchors))
        reference_points = (pos_anchors)
        init_reference_out = reference_points
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_norm[head_idx](
                self.aux_pos_trans[head_idx](self.get_proposal_pos_embed(topk_coords_unact)))
            query = pos_trans_out
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](self.pos_feats_trans[head_idx](pos_feats))

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        inter_states, inter_references, attn_weights_list = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references

        return inter_states, inter_references_out

@ATTENTION.register_module()
class TrackingMultiheadAttention(MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_attn_weights = None

    def forward(self, query, key=None, value=None, *args, **kwargs):
        # Only pass specific arguments to self.attn (nn.MultiheadAttention)
        # to avoid errors with extra arguments passed by Co-DETR transformer.
        attn_kwargs = {}
        if 'key_padding_mask' in kwargs:
            attn_kwargs['key_padding_mask'] = kwargs['key_padding_mask']
        if 'attn_mask' in kwargs:
            attn_kwargs['attn_mask'] = kwargs['attn_mask']
        
        if key is None:
            key = query
        if value is None:
            value = key

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, attn_weights = self.attn(
            query,
            key,
            value,
            *args,
            need_weights=True,
            average_attn_weights=True, # We want averaged for simplifying the metric
            **attn_kwargs)
        
        self.last_attn_weights = attn_weights.detach()
        
        if self.batch_first:
            out = out.transpose(0, 1)
            
        return out


@TRANSFORMER_LAYER.register_module()
class TrackingDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(len(self.attentions))]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(len(self.attentions))
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == len(self.attentions)

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query_select = query + query_pos if query_pos is not None else query
                temp_attn_mask = attn_masks[attn_index]
                query = self.attentions[attn_index](
                    query_select,
                    temp_key,
                    temp_value,
                    attn_mask=temp_attn_mask,
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'cross_attn':
                temp_key = temp_value = value
                query_select = query + query_pos if query_pos is not None else query
                key_select = temp_key + key_pos if key_pos is not None else temp_key
                temp_attn_mask = attn_masks[attn_index]
                query = self.attentions[attn_index](
                    query_select,
                    key_select,
                    temp_value,
                    attn_mask=temp_attn_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
