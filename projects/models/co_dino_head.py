# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh, multi_apply,
                        reduce_mean, bbox_overlaps)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.builder import HEADS
from mmcv.ops import batched_nms
from projects.models import CoDeformDETRHead
from projects.models.query_denoising import build_dn_generator
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger

@HEADS.register_module()
class CoDINOHead(CoDeformDETRHead):

    def __init__(self,
                 *args,
                 num_query=900,
                 num_track_queries=0,
                 spawn_score_thresh=0.5,
                 miss_tolerance=1,
                 track_loss_weight=1.0,
                 query_init_checkpoint=None,
                 dn_cfg=None,
                 transformer=None,
                 **kwargs):

        if 'two_stage_num_proposals' in transformer:
            assert transformer['two_stage_num_proposals'] == num_query, \
                'two_stage_num_proposals must be equal to num_query for DINO'
        else:
            transformer['two_stage_num_proposals'] = num_query
        self.num_track_queries = num_track_queries
        self.spawn_score_thresh = spawn_score_thresh
        self.miss_tolerance = miss_tolerance
        self.track_loss_weight = track_loss_weight
        self.query_init_checkpoint = query_init_checkpoint

        super(CoDINOHead, self).__init__(
            *args, num_query=num_query, transformer=transformer, **kwargs)
        


        assert self.as_two_stage, \
            'as_two_stage must be True for DINO'
        assert self.with_box_refine, \
            'with_box_refine must be True for DINO'
        self._init_layers()
        self.init_denoising(dn_cfg)

    def _init_layers(self):
        super()._init_layers()
        self.query_embedding = None
        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates [Unknown] class. However, the embedding of unknown class
        # is not used in the original DINO
        self.label_embedding = nn.Embedding(self.cls_out_channels,
                                            self.embed_dims)
        # Learnable embedding for empty/new track slots
        # NOTE: track_embed is initialized randomly by default (nn.Embedding behaviour)
        if self.num_track_queries > 0:
            self.track_embed = nn.Embedding(self.num_track_queries, self.embed_dims)
            # Track state buffer: [matched_id, miss_count] (-1 = empty)
            # We don't register this as a buffer because it's per-sequence and handled via cache,
            # but we can keep a template or helper if needed.

        self.downsample = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.embed_dims)
        )
        
        
    def init_weights(self):
        super().init_weights()
        # [Custom Checkpoint Loading]
        if self.query_init_checkpoint is not None:
            logger = get_root_logger()
            rank, _ = get_dist_info()
            if rank == 0:
                logger.info(f"[CoDINOHead] Loading query_init_checkpoint: {self.query_init_checkpoint}")
            
            try:
                ckpt = torch.load(self.query_init_checkpoint, map_location='cpu')
                if 'state_dict' in ckpt:
                    ckpt = ckpt['state_dict']
                
                # Potential keys for query_embed in standard Co-DETR/DINO checkpoints
                key_cands = [
                    'query_head.transformer.query_embed.weight',
                    'query_head.query_embedding.weight',
                    'transformer.query_embed.weight'
                ]
                
                loaded_weight = None
                for key in key_cands:
                    if key in ckpt:
                        loaded_weight = ckpt[key]
                        if rank == 0: logger.info(f"[CoDINOHead] Found queries in checkpoint key: {key}")
                        break
                
                if loaded_weight is not None:
                    # Target: self.transformer.query_embed
                    if hasattr(self.transformer, 'query_embed'):
                        target_embed = self.transformer.query_embed
                        N_ckpt = loaded_weight.size(0)
                        N_model = target_embed.weight.size(0)
                        
                        if rank == 0:
                             logger.info(f"[CoDINOHead] Checkpoint queries: {N_ckpt}, Model New Object queries: {N_model}")
                        
                        # Handle size mismatch (slice)
                        if N_ckpt >= N_model:
                             with torch.no_grad():
                                 target_embed.weight.copy_(loaded_weight[:N_model])
                             if rank == 0:
                                 logger.info(f"[CoDINOHead] Successfully loaded first {N_model} queries from checkpoint to transformer.query_embed.")
                        else:
                             if rank == 0:
                                 logger.warning(f"[CoDINOHead] Checkpoint has FEWER queries ({N_ckpt}) than model ({N_model}). Skipping load to avoid undefined behavior.")
                    else:
                        if rank == 0:
                            logger.warning("[CoDINOHead] self.transformer does not have 'query_embed' attribute.")
                else:
                    if rank == 0:
                         logger.warning(f"[CoDINOHead] Could not find any of keys {key_cands} in checkpoint.")
            except Exception as e:
                 if rank == 0:
                     logger.error(f"[CoDINOHead] Error loading query checkpoint: {e}")

    def init_denoising(self, dn_cfg):
        if dn_cfg is not None:
            dn_cfg['num_classes'] = self.num_classes
            dn_cfg['num_queries'] = self.num_query
            dn_cfg['hidden_dim'] = self.embed_dims
        self.dn_generator = build_dn_generator(dn_cfg)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        assert proposal_cfg is None, '"proposal_cfg" must be None'
        assert self.dn_generator is not None, '"dn_cfg" must be set'
        dn_label_query, dn_bbox_query, attn_mask, dn_meta = \
            self.dn_generator(gt_bboxes, gt_labels,
                              self.label_embedding, img_metas)
        outs = self(x, img_metas, dn_label_query, dn_bbox_query, attn_mask, 
                   gt_bboxes=gt_bboxes, gt_labels=gt_labels)
        head_outputs = outs[:5]
        attn_metrics = outs[-1] # Assuming it's the last element now
        if gt_labels is None:
            loss_inputs = head_outputs + (gt_bboxes, img_metas, dn_meta)
        else:
            loss_inputs = head_outputs + (gt_bboxes, gt_labels, img_metas, dn_meta)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        # Add attention metrics to losses for logging
        if attn_metrics is not None and isinstance(attn_metrics, dict):
            losses.update(attn_metrics)
            
        enc_outputs = head_outputs[-1]
        with torch.no_grad():
            tmp_results_list = self.get_bboxes(*head_outputs, img_metas=img_metas, rescale=False, with_nms=False)
            results_list = [res[0] for res in tmp_results_list]
        return losses, enc_outputs, results_list

    def forward(self,
                mlvl_feats,
                img_metas,
                dn_label_query=None,
                dn_bbox_query=None,
                attn_mask=None,
                gt_bboxes=None,
                gt_labels=None):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        prev_query_feats_list, prev_reference_points_list, prev_valid_lengths = (
            self._collect_prev_queries(img_metas))

        transformer_out = self.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                dn_label_query,
                dn_bbox_query,
                attn_mask,
                prev_query_feats_list=prev_query_feats_list,
                prev_reference_points_list=prev_reference_points_list,
                prev_query_valid_lens=prev_valid_lengths,
                return_decoder_cache=True,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
            )

        # Unpack the results including attn_metrics
        if len(transformer_out) == 7:
            hs, inter_references, topk_score, topk_anchor, enc_outputs, decoder_cache, attn_metrics = transformer_out
        else:
             # Fallback if modification failed or mismatch
            hs, inter_references, topk_score, topk_anchor, enc_outputs, decoder_cache = transformer_out
            attn_metrics = {}

        outs = []
        num_level = len(mlvl_feats)
        start = 0
        for lvl in range(num_level):
            bs, c, h, w = mlvl_feats[lvl].shape
            end = start + h*w
            feat = enc_outputs[start:end].permute(1, 2, 0).contiguous()
            start = end
            outs.append(feat.reshape(bs, c, h, w))
        outs.append(self.downsample(outs[-1]))

        hs = hs.permute(0, 2, 1, 3)

        if dn_label_query is not None and dn_label_query.size(1) == 0:
            # NOTE: If there is no target in the image, the parameters of
            # label_embedding won't be used in producing loss, which raises
            # RuntimeError when using distributed mode.
            hs[0] += self.label_embedding.weight[0, 0] * 0.0

        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        
        # Pass outputs to update cache
        self._update_sequence_cache(img_metas, decoder_cache, 
                                    outputs_classes, outputs_coords)

        return outputs_classes, outputs_coords, topk_score, topk_anchor, outs, decoder_cache, attn_metrics

    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_topk_scores,
             enc_topk_anchors,
             enc_outputs, 
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             dn_meta=None,
             gt_bboxes_ignore=None):
        # assert gt_bboxes_ignore is None, \
        #     f'{self.__class__.__name__} only supports ' \
        #     f'for gt_bboxes_ignore setting to None.'

        loss_dict = dict()

        # extract denoising and matching part of outputs
        all_cls_scores, all_bbox_preds, dn_cls_scores, dn_bbox_preds = \
            self.extract_dn_outputs(all_cls_scores, all_bbox_preds, dn_meta)

        if enc_topk_scores is not None:
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = \
                self.loss_single(enc_topk_scores, enc_topk_anchors,
                                 gt_bboxes_list, gt_labels_list,
                                 img_metas, gt_bboxes_ignore)

            # collate loss from encode feature maps
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_iou'] = enc_losses_iou
        
        # [Logging] Track vs New detection stats
        if self.num_track_queries > 0:
            with torch.no_grad():
                # Last decoder layer scores
                last_scores = all_cls_scores[-1].sigmoid() # [B, Q_all, C]
                max_scores, _ = last_scores.max(-1) # [B, Q_all]
                
                # Split
                N_new = self.num_query
                scores_new = max_scores[:, :N_new]
                scores_track = max_scores[:, N_new : N_new + self.num_track_queries]
                
                # Count detections > threshold (e.g. 0.3 matching visualization)
                thresh = 0.3
                n_new_det = (scores_new > thresh).float().sum()
                n_track_det = (scores_track > thresh).float().sum()
                total_det = n_new_det + n_track_det
                
                # Avoid division by zero
                ratio_track = n_track_det / (total_det + 1e-6)
                
                # Add to loss_dict (keys without 'loss' are logged but not optimized)
                loss_dict['stat_num_new_det'] = n_new_det / max(1, len(scores_new)) # Normalize by batch? No, just average count per batch via reduce_mean later
                loss_dict['stat_num_track_det'] = n_track_det / max(1, len(scores_new))
                loss_dict['stat_track_ratio'] = ratio_track

        # calculate loss from all decoder layers
        num_dec_layers = len(all_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        # collate loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_iou'] = losses_iou[-1]

        # collate loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
            num_dec_layer += 1

        if dn_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_meta = [dn_meta for _ in img_metas]
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                dn_cls_scores, dn_bbox_preds, gt_bboxes_list, gt_labels_list,
                img_metas, dn_meta)
            # collate denoising loss
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            loss_dict['dn_loss_iou'] = dn_losses_iou[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1],
                    dn_losses_iou[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                loss_dict[f'd{num_dec_layer}.dn_loss_iou'] = loss_iou_i
                num_dec_layer += 1

        return loss_dict

    def loss_dn(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
                gt_labels_list, img_metas, dn_meta):
        num_dec_layers = len(dn_cls_scores)
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        dn_meta_list = [dn_meta for _ in range(num_dec_layers)]
        return multi_apply(self.loss_dn_single, dn_cls_scores, dn_bbox_preds,
                           all_gt_bboxes_list, all_gt_labels_list,
                           img_metas_list, dn_meta_list)

    def loss_dn_single(self, dn_cls_scores, dn_bbox_preds, gt_bboxes_list,
                       gt_labels_list, img_metas, dn_meta):
        num_imgs = dn_cls_scores.size(0)
        bbox_preds_list = [dn_bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_dn_target(bbox_preds_list, gt_bboxes_list,
                                             gt_labels_list, img_metas,
                                             dn_meta)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        
        pred_dtype = dn_bbox_preds.dtype
        for i in range(len(bbox_targets_list)):
            if bbox_targets_list[i].dtype != pred_dtype:
                bbox_targets_list[i] = bbox_targets_list[i].to(pred_dtype)
            if bbox_weights_list[i].dtype != pred_dtype:
                bbox_weights_list[i] = bbox_weights_list[i].to(pred_dtype)

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = dn_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = \
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if len(cls_scores) > 0:
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = dn_bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)

            ov = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)

            scores = ov.new_zeros(labels.shape)
            scores[pos_inds] = ov

            loss_cls = self.loss_cls(
                cls_scores, (labels, scores),
                weight=label_weights,
                avg_factor=cls_avg_factor)

        else:
            loss_cls = torch.zeros(  # TODO: How to better return zero loss
                1,
                dtype=cls_scores.dtype,
                device=cls_scores.device)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, dn_bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def get_dn_target(self, dn_bbox_preds_list, gt_bboxes_list, gt_labels_list,
                      img_metas, dn_meta):
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_dn_target_single,
                                      dn_bbox_preds_list, gt_bboxes_list,
                                      gt_labels_list, img_metas, dn_meta)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_dn_target_single(self, dn_bbox_pred, gt_bboxes, gt_labels,
                              img_meta, dn_meta):
        num_groups = dn_meta['num_dn_group']
        pad_size = dn_meta['pad_size']
        assert pad_size % num_groups == 0
        single_pad = pad_size // num_groups
        num_bboxes = dn_bbox_pred.size(0)

        if len(gt_labels) > 0:
            t = torch.range(0, len(gt_labels) - 1).long().cuda()
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = (torch.tensor(range(num_groups)) *
                        single_pad).long().cuda().unsqueeze(1) + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = torch.tensor([]).long().cuda()
        neg_inds = pos_inds + single_pad // 2

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(dn_bbox_pred)
        bbox_weights = torch.zeros_like(dn_bbox_pred)
        bbox_weights[pos_inds] = 1.0
        img_h, img_w, _ = img_meta['img_shape']

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = dn_bbox_pred.new_tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0)
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)

        if gt_bboxes_targets.dtype != bbox_targets.dtype:
            gt_bboxes_targets = gt_bboxes_targets.to(bbox_targets.dtype)

        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    @staticmethod
    def extract_dn_outputs(all_cls_scores, all_bbox_preds, dn_meta):
        if dn_meta is not None:
            denoising_cls_scores = all_cls_scores[:, :, :
                                                  dn_meta['pad_size'], :]
            denoising_bbox_preds = all_bbox_preds[:, :, :
                                                  dn_meta['pad_size'], :]
            matching_cls_scores = all_cls_scores[:, :, dn_meta['pad_size']:, :]
            matching_bbox_preds = all_bbox_preds[:, :, dn_meta['pad_size']:, :]
        else:
            denoising_cls_scores = None
            denoising_bbox_preds = None
            matching_cls_scores = all_cls_scores
            matching_bbox_preds = all_bbox_preds
        return (matching_cls_scores, matching_bbox_preds, denoising_cls_scores,
                denoising_bbox_preds)

    def forward_aux(self, mlvl_feats, img_metas, aux_targets, head_idx):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        aux_coords, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks = aux_targets
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        hs, inter_references = self.transformer.forward_aux(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    aux_coords,
                    pos_feats=aux_feats,
                    reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
                    return_encoder_output=True,
                    attn_masks=attn_masks,
                    head_idx=head_idx
            )

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords, \
                None, None

    def _collect_prev_queries(self, img_metas):
        """
        Collect fixed-size track queries from cache or initialize them.
        Returns:
            prev_query_feats_list: List of [N_track, C] tensors
            prev_reference_points_list: List of [N_track, 4] tensors 
            prev_valid_lengths: List of ints (always N_track if tracking enabled)
        """
        if self.num_track_queries <= 0:
            return super()._collect_prev_queries(img_metas)

        device = self.label_embedding.weight.device
        dtype = self.label_embedding.weight.dtype
        
        prev_query_feats_list = []
        prev_reference_points_list = []
        prev_valid_lengths = []

        # [DDP Fix] Always touch track_embed to keep it in the graph
        # Prevents "Unused parameter" error without using find_unused_parameters=True (which breaks checkpointing)
        dummy_tensor = self.track_embed.weight.sum() * 0.0

        for meta in img_metas:
            seq_key = self._get_sequence_key(meta)
            reset = self._should_reset_sequence(meta)
            
            cached = None
            if not reset and seq_key is not None:
                cached = self._prev_decoder_cache.get(seq_key, None)
            
            if cached is None:
                # Initialize empty tracks [N_track, C]
                feats = self.track_embed.weight.clone()
                # Refs: [N_track, 4] -> Center [0.5, 0.5, 1.0, 1.0]
                refs = torch.tensor([0.5, 0.5, 1.0, 1.0], device=device, dtype=dtype).unsqueeze(0).repeat(self.num_track_queries, 1)
                
                # Track Info: [id, miss_count] (-1 = empty)
                track_info = torch.full((self.num_track_queries, 2), -1, device=device, dtype=torch.long)
                track_info[:, 1] = 0 # miss count
                
                if seq_key is not None:
                    # Clear old cache if it existed but we reset
                    self._prev_decoder_cache.pop(seq_key, None)
            else:
                feats = cached['query_feats'].to(device).to(dtype)
                # [DDP Fix] Add dummy dependency
                feats = feats + dummy_tensor
                
                refs = cached['reference_points'].to(device).to(dtype)
                track_info = cached['track_info'].to(device)

            prev_query_feats_list.append(feats)
            prev_reference_points_list.append(refs)
            prev_valid_lengths.append(self.num_track_queries)
            
            meta['track_info'] = track_info

        return prev_query_feats_list, prev_reference_points_list, prev_valid_lengths

    def _update_sequence_cache(self, img_metas, decoder_cache, outputs_classes=None, outputs_coords=None):
        if self.num_track_queries <= 0:
            return super()._update_sequence_cache(img_metas, decoder_cache)
        
        if decoder_cache is None or outputs_classes is None or outputs_coords is None:
            return

        batch_cls = outputs_classes[-1]  # [B, N_all, C]
        batch_box = outputs_coords[-1]   # [B, N_all, 4]
        
        # Transformer outputs
        # [B, N_all, C]
        cached_feats = decoder_cache.get('query_feats') 
        # [B, N_all, 4] (if sigmoid)
        cached_refs = decoder_cache.get('reference_points')
        
        if cached_feats is None or cached_refs is None:
            return

        N_all = cached_feats.size(1)
        N_new = self.num_query
        N_track = self.num_track_queries
        
        # If sizes mismatch (e.g. transformer didn't include tracks), we might be in Frame 1
        # where only New queries existed.
        has_tracks_in_output = (N_all >= N_new + N_track)
        
        if N_all < N_new:
             # Critical error, less than new queries?
             return

        # NOTE: Cached features are from PREVIOUS query interaction (input to transformer)
        # We need "Output of Transformer" for NEXT frame?
        # No, "Output of Transformer" = "Input to Next Frame".
        # Correct.

        for idx, meta in enumerate(img_metas):
            seq_key = self._get_sequence_key(meta)
            if seq_key is None:
                continue
            
            # --- 1. Slicing ---
            # Transformer appends tracks: [Q_new, Q_track]
            
            # Feats for NEXT frame
            out_new_feats = cached_feats[idx, :N_new]
            out_new_refs = cached_refs[idx, :N_new]
            
            out_track_feats = None
            out_track_refs = None
            if has_tracks_in_output:
                out_track_feats = cached_feats[idx, N_new : N_new + N_track]
                out_track_refs = cached_refs[idx, N_new : N_new + N_track]
            
            # Scores for CURRENT frame
            # scores are usually [N, NumClasses]. We need max score?
            pred_cls = batch_cls[idx] # [N_all, C]
            pred_box = batch_box[idx] # [N_all, 4]
            
            score_new, _ = pred_cls[:N_new].max(-1) # sigmoid score or logit?
            if self.loss_cls.use_sigmoid:
                score_new = score_new.sigmoid()
            
            score_track = None
            if has_tracks_in_output:
                score_track, _ = pred_cls[N_new : N_new + N_track].max(-1)
                if self.loss_cls.use_sigmoid:
                     score_track = score_track.sigmoid()
            else:
                 # Dummy scores for missing tracks (all should be inactive anyway)
                 score_track = torch.zeros(N_track, device=pred_cls.device)
            
            # --- 2. Track Lifecycle ---
            
            # Retrieve current track info (loaded in _collect)
            # [id, miss_count]
            # We need to construct the NEXT track info.
            current_track_info = meta.get('track_info', None)
            if current_track_info is None:
                current_track_info = torch.full((N_track, 2), -1, device=batch_cls.device)
            
            next_track_info = current_track_info.clone()
            
            if out_track_feats is not None:
                next_track_feats = out_track_feats.clone()
                next_track_refs = out_track_refs.clone()
            else:
                # No track outputs? Initialize to defaults (will be overwritten by spawns or remain inactive)
                next_track_feats = self.track_embed.weight.clone()
                next_track_refs = torch.tensor([0.5, 0.5, 1.0, 1.0], device=batch_cls.device).unsqueeze(0).repeat(N_track, 1)
            
            # Identify active tracks
            active_mask = current_track_info[:, 0] >= 0
            
            # A. Update Active Tracks
            # Simple Logic: If score > thresh, keep. Else miss.
            # (Ideally matching to GT if training, but stick to simple for now)
            
            # Check hits
            is_hit = score_track > self.spawn_score_thresh # Reuse spawn thresh or add track_thresh
            
            # If hit -> miss_count = 0. Feature updated (already taken from out_track_feats).
            next_track_info[active_mask & is_hit, 1] = 0
            
            # If miss -> miss_count += 1. 
            # Feature? Keep OLD feature? Or use new prediction? 
            # If it's a miss, the prediction is likely background/garbage.
            # Better to strictly KEEP OLD FEATURE if missed to avoid drift into background.
            # But we need access to OLD feature. Is it in `out_track_feats`? 
            # `out_track_feats` is the refined output.
            # We need `prev_query_feats` which was INPUT.
            # It's not readily available here unless we stored it in meta or cache.
            # We can read from `self._prev_decoder_cache` (previous state).
            prev_cache = self._prev_decoder_cache.get(seq_key)
            if prev_cache is not None:
                prev_feats_input = prev_cache['query_feats'].to(batch_cls.device)
                prev_refs_input = prev_cache['reference_points'].to(batch_cls.device)
                
                # Restore features for misses
                miss_indices = active_mask & (~is_hit)
                if miss_indices.any():
                    next_track_info[miss_indices, 1] += 1
                    # Restore previous features/refs to avoid polluting with background
                    next_track_feats[miss_indices] = prev_feats_input[miss_indices]
                    next_track_refs[miss_indices] = prev_refs_input[miss_indices]
            
            # B. Retire Dead Tracks
            dead_mask = next_track_info[:, 1] > self.miss_tolerance
            next_track_info[dead_mask, 0] = -1 # ID -> -1 (empty)
            next_track_info[dead_mask, 1] = 0
            # Reset features for dead slots to learnable embedding?
            if dead_mask.any():
                 next_track_feats[dead_mask] = self.track_embed.weight[dead_mask]
                 # Reset refs to center?
                 # Assuming refs are sigmoid:
                 next_track_refs[dead_mask] = torch.tensor([0.5, 0.5, 1.0, 1.0], device=batch_cls.device).to(batch_cls.dtype)

            # C. Spawn New Tracks
            # Find empty slots
            empty_mask = next_track_info[:, 0] < 0
            num_empty = empty_mask.sum().item()
            
            if num_empty > 0:
                # Find high confidence new detections
                # Simple NMS-like or just top-k?
                # Greedy: Top-k scores
                top_vals, top_inds = torch.topk(score_new, k=min(num_empty, N_new))
                
                # Filter by threshold
                valid_new = top_vals > self.spawn_score_thresh
                spawn_inds = top_inds[valid_new]
                
                if len(spawn_inds) > 0:
                    # Fill empty slots
                    empty_indices = torch.nonzero(empty_mask, as_tuple=True)[0]
                    num_spawn = min(len(spawn_inds), len(empty_indices))
                    
                    fill_slots = empty_indices[:num_spawn]
                    source_inds = spawn_inds[:num_spawn]
                    
                    next_track_feats[fill_slots] = out_new_feats[source_inds]
                    next_track_refs[fill_slots] = out_new_refs[source_inds]
                    
                    # Assign New IDs? 
                    # If we don't have global ID management, we just mark as "Active" (id=1, or increments).
                    # For now just set ID=1 to mark used.
                    next_track_info[fill_slots, 0] = 1 
                    next_track_info[fill_slots, 1] = 0

            # --- 3. Save to Cache ---
            self._prev_decoder_cache[seq_key] = {
                'query_feats': next_track_feats.detach(),
                'reference_points': next_track_refs.detach(),
                'track_info': next_track_info,
                'valid_length': N_track
            }


    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    img_metas,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        with torch.cuda.amp.autocast(enabled=False):
            cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           img_metas, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        pred_dtype = bbox_preds.dtype
        for i in range(len(bbox_targets_list)):
            if bbox_targets_list[i].dtype != pred_dtype:
                bbox_targets_list[i] = bbox_targets_list[i].to(pred_dtype)
            if bbox_weights_list[i].dtype != pred_dtype:
                bbox_weights_list[i] = bbox_weights_list[i].to(pred_dtype)

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)

        ov = bbox_overlaps(
            pos_decode_bbox_pred.detach(),
            pos_decode_bbox_targets,
            is_aligned=True)

        # scores must match ov.dtype to avoid index-assign dtype error
        scores = ov.new_zeros(labels.shape)      # dtype = ov.dtype
        scores[pos_inds] = ov
        loss_cls = self.loss_cls(
            cls_scores, (labels, scores),
            weight=label_weights,
            avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls, loss_bbox, loss_iou

    def loss_single_aux(self,
                        cls_scores,
                        bbox_preds,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        img_metas,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        num_q = cls_scores.size(1)
        try:
            labels = labels.reshape(num_imgs * num_q)
            label_weights = label_weights.reshape(num_imgs * num_q)
            bbox_targets = bbox_targets.reshape(num_imgs * num_q, 4)
            bbox_weights = bbox_weights.reshape(num_imgs * num_q, 4)
        except:
            return cls_scores.mean()*0, cls_scores.mean()*0, cls_scores.mean()*0

        bg_class_ind = self.num_classes
        num_total_pos = len(((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1))
        num_total_neg = num_imgs*num_q - num_total_pos

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        scores = label_weights.new_zeros(labels.shape)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
        scores[pos_inds] = bbox_overlaps(
            pos_decode_bbox_pred.detach(),
            pos_decode_bbox_targets,
            is_aligned=True)
        loss_cls = self.loss_cls(
            cls_scores, (labels, scores),
            weight=label_weights,
            avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            img_h, img_w, _ = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls*self.lambda_1, loss_bbox*self.lambda_1, loss_iou*self.lambda_1
