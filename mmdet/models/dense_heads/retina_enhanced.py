# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_conv_layer, bias_init_with_prob, constant_init, is_norm,normal_init
from ..builder import HEADS
from .anchor_head import AnchorHead
from timm.models.layers import trunc_normal_
from mmdet.core import multi_apply
from .GCA import GCA
from .LDCA import LDCA
from .BACA import BACA

@HEADS.register_module()
class RetinaHeadEnhanced(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHeadEnhanced, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        dcn_cfg=dict(type='DCNv2', deform_groups=1)
        self.stem = build_conv_layer(
            dcn_cfg,
            self.in_channels,
            self.feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            bias=False)

        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)

        self.Csw_block1 = GCA(self.feat_channels, latent_ratio=16)
        self.Csw_block2 = GCA(self.feat_channels, latent_ratio=16)
        self.Csw_block3 = GCA(self.feat_channels, latent_ratio=16)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_convs_2 = nn.ModuleList()
        self.reg_convs_2 = nn.ModuleList()
        self.cls_convs_3 = nn.ModuleList()
        self.reg_convs_3 = nn.ModuleList()
        dim_reduction = 4
        mlp_ratio=3
        ks=3
        self.cls_convs.append(
            LDCA(self.feat_channels, ks, dim_reduction=dim_reduction, mlp_ratio=mlp_ratio)
        )
        self.cls_convs_2.append(
            LDCA(self.feat_channels, ks, dim_reduction=dim_reduction, mlp_ratio=mlp_ratio)
        )
        self.reg_convs.append(
            LDCA(self.feat_channels, ks, dim_reduction=dim_reduction, mlp_ratio=mlp_ratio)
        )
        self.reg_convs_2.append(
            LDCA(self.feat_channels, ks, dim_reduction=dim_reduction, mlp_ratio=mlp_ratio)
        )

        self.cls_convs_3.append(
            LDCA(self.feat_channels, ks, dim_reduction=dim_reduction, mlp_ratio=mlp_ratio)
        )
        self.reg_convs_3.append(
            LDCA(self.feat_channels, ks, dim_reduction=dim_reduction, mlp_ratio=mlp_ratio)
        )

        self.dcim_ks = 3
        self.shrink_ratio = 1.
        self.gradient_mul = 0.1
        self.strides = [8, 16, 32, 64, 128]
        self.retina_cls = BACA(self.feat_channels, ks=self.dcim_ks, num_class=self.cls_out_channels, num_anchor=self.num_anchors)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)

            ####add by davina
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # Use prior in model initialization to improve stability
        bias_cls = bias_init_with_prob(0.01)
        # torch.nn.init.constant_(self.retina_cls.bias, bias_cls)
        torch.nn.init.constant_(self.retina_cls.cls_conv.bias, bias_cls)
        nn.init.constant_(self.retina_cls.conv_offset[-1].weight, 0)
        nn.init.constant_(self.retina_cls.conv_offset[-1].bias, 0)

    def forward_single(self, x, anchors, stride, img_meta):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        x=self.stem(x)
        x = self.Csw_block1(x)
        x = self.Csw_block2(x)
        x = self.Csw_block3(x)

        cls_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_feat_ = cls_conv(cls_feat,reg_feat)
        for reg_conv in self.reg_convs:
            reg_feat_ = reg_conv(reg_feat,cls_feat)

        for cls_conv in self.cls_convs_2:
            cls_feat__ = cls_conv(cls_feat_,reg_feat_)
        for reg_conv in self.reg_convs_2:
            reg_feat__ = reg_conv(reg_feat_,cls_feat_)
        for cls_conv in self.cls_convs_3:
            cls_feat = cls_conv(cls_feat__, reg_feat__)
        for reg_conv in self.reg_convs_3:
            reg_feat = reg_conv(reg_feat__, cls_feat__)

        bbox_pred = self.retina_reg(reg_feat)

        B, C, H, W = x.shape
        anchors = anchors.reshape(-1, 4)
        bbox_pred_grad_mul = (1 - self.gradient_mul) * bbox_pred.detach() + self.gradient_mul * bbox_pred
        bbox_pred_grad_mul = bbox_pred_grad_mul.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred_decode = self.bbox_coder.decode(anchors, bbox_pred_grad_mul)
        bbox_pred_decode = bbox_pred_decode.reshape(B, H, W, self.num_anchors, 4)
        decode_offsets = self.decode_bbox_offset(bbox_pred_decode) / stride
        decode_offsets = decode_offsets.reshape(B, -1, H, W)
        cls_score = self.retina_cls(cls_feat, decode_offsets)

        return cls_score, bbox_pred

    def decode_bbox_offset(self, bbox_pred):
        B, H, W, _, _ = bbox_pred.shape
        bbox_pred = bbox_pred.permute(0, 3, 4, 1, 2)
        decode_offset = bbox_pred.new_zeros(B, self.num_anchors, self.dcim_ks ** 2, 2, H, W)
        x1 = bbox_pred[:, :, 0, :, :]
        y1 = bbox_pred[:, :, 1, :, :]
        x2 = bbox_pred[:, :, 2, :, :]
        y2 = bbox_pred[:, :, 3, :, :]
        w, h = x2 - x1, y2 - y1

        if self.shrink_ratio != 1.:
            w_, h_ = self.shrink_ratio * w, self.shrink_ratio * h
            x1 = x1 + (w - w_) / 2.
            y1 = y1 + (h - h_) / 2.
            w = w_
            h = h_
        for row in range(self.dcim_ks):
            for col in range(self.dcim_ks):
                sampling_index = row * self.dcim_ks + col
                decode_offset[..., sampling_index, 0, :, :] = y1 + row * h / (self.dcim_ks - 1)
                decode_offset[..., sampling_index, 1, :, :] = x1 + col * w / (self.dcim_ks - 1)

        return decode_offset

    def forward(self, feats, img_metas=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in feats]
        device = feats[0].device

        # anchor_list -> [img1[level5], img2[level5]]
        anchor_list, _ = self.get_anchors(featmap_sizes, img_metas, device=device)
        anchor_fmfirst_list = []
        for i in range(len(anchor_list[0])):
            temp_list = []
            for j in range(len(anchor_list)):
                temp = anchor_list[j][i]
                temp_list.append(temp)
            anchor_fmfirst_list.append(torch.stack(temp_list, dim=0))
        img_metas_list = [img_metas for i in range(len(self.strides))]

        return multi_apply(self.forward_single, feats, anchor_fmfirst_list, self.strides, img_metas_list)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x, img_metas)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        outs = self.forward(feats, img_metas)
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        if img_metas is None:
            img_metas = [dict(pad_shape=(0, 0))]
        num_imgs = len(img_metas)
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list