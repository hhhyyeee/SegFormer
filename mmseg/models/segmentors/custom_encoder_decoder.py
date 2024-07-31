from collections import OrderedDict
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models.utils import freeze
from mmseg.utils import get_root_logger
from tools.get_param_count import count_parameters

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class OthersEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        **cfg
        ):
        super(OthersEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        a=1
        num_module = 4
        self.decodesc_flag = "decoder_custom" in backbone

        self.freeze_backbone = cfg.get("freeze_backbone", None)
        self.freeze_decode_head = cfg.get("freeze_decode_head", None)
        if (self.freeze_backbone is not None) and (self.freeze_backbone == True):
            freeze(self.backbone)
        if (self.freeze_decode_head is not None) and (self.freeze_decode_head == True):
            freeze(self.decode_head)
        count_parameters(self.backbone)
        count_parameters(self.decode_head)
        a=1
    
    def get_main_model(self):
        return self.main_model

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        if pretrained is None: return
        logger = get_root_logger()
        state_dict = _load_checkpoint(pretrained, logger=logger, map_location="cpu")
        backbone_state_dict = OrderedDict({k.replace("backbone.", ""): v for k, v in state_dict["state_dict"].items() if "backbone" in k})
        decode_head_state_dict = OrderedDict({k.replace("decode_head.", ""): v for k, v in state_dict["state_dict"].items() if "decode_head" in k})
        a=1

        # self.backbone.init_weights()
        self.backbone.load_state_dict(backbone_state_dict, False)
        # self.decode_head.load_state_dict(decode_head_state_dict, True)
        # self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()

        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat_decodesc(self, img):
        a=1
        x, c = self.backbone(img)
        a=1
        if self.with_neck:
            x = self.neck(x)
        
        return x, c
    
    def _decode_head_forward_train_decodesc(self, x, c, img_metas,
                                            gt_semantic_seg, seg_weight=None):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, c, img_metas,
                                                     gt_semantic_seg,
                                                     train_cfg=self.train_cfg,
                                                     seg_weight=seg_weight)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses
    
    def _decode_head_forward_test_decodesc(self, x, c, img_metas):
        seg_logits = self.decode_head.forward_test(x, c, img_metas, self.test_cfg)
        return seg_logits

    def forward_train(self, img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=False):
        a=1
        if self.decodesc_flag:
            x, c = self.extract_feat_decodesc(img)
            a=1
        else:
            x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x

        if self.decodesc_flag:
            a=1
            loss_decode = self._decode_head_forward_train_decodesc(x, c, img_metas,
                                                                   gt_semantic_seg, seg_weight)
        else:
            loss_decode = self._decode_head_forward_train(x, img_metas,
                                                          gt_semantic_seg, seg_weight)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg, seg_weight)
            losses.update(loss_aux)

        return losses
    
    def encode_decode(self, img, img_metas):
        a=1
        if self.decodesc_flag:
            x, c = self.extract_feat_decodesc(img)
            out = self._decode_head_forward_test_decodesc(x, c, img_metas)
        else:
            x = self.extract_feat(img)
            out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def entropy_prediction(self, img):
        x = self.extract_feat(img)

        entr, conf = self.decode_head.calculate_entropy(x)

        return {f"confidence": conf, "entropy": entr}


