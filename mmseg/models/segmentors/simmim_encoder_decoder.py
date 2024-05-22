import torch
import torch.nn.functional as F
import torch.distributed as dist

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.utils import freeze

from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

from typing import List
import numpy as np
from collections import OrderedDict


class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):

        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        assert isinstance(input_size, tuple)

        H, W = input_size
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert H % self.mask_patch_size == 0
        assert W % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = np.zeros(2, dtype=int)
        self.rand_size[0] = self.input_size[0] // self.mask_patch_size
        self.rand_size[1] = self.input_size[1] // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size[0] * self.rand_size[1]
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, bulk=False):
        if not bulk:
            mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        else:
            pass
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size[0], self.rand_size[1]))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        mask = torch.from_numpy(mask).unsqueeze(0) #!DEBUG

        return mask


@SEGMENTORS.register_module()
class SimMIMEncoderDecoder(EncoderDecoder):
    """
    - base task: Semantic Segmentation
    - masked image modeling
    """
    def __init__(
        self,
        backbone,
        decode_head,
        neck=None,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        mim_cfg=None,
        init_cfg=None,
        **cfg
        ):
        super(SimMIMEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        a=1

        self.recon_neck = builder.build_neck(cfg["recon_neck"])
        self.recon_head = builder.build_head(cfg["recon_head"])

        self.mask_generator = MaskGenerator(
            input_size=backbone["img_size"], #720,
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=0.6)

        a=1
        self.test_mask_generator = MaskGenerator(
            input_size=(1024, 2048),
            mask_patch_size=32,
            model_patch_size=4,
            mask_ratio=0.6
        )

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        # simmim
        a=1

        # segmentation
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)


    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        a=1
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)

        a=1
        rec_pred = self.reconstruct(img)

        # #!DEBUG
        # prob, pred = seg_logit.max(dim=1)
        # prob = [prob.cpu().numpy()]
        # pred = [pred.cpu().numpy()]

        return seg_pred #, prob, pred


    def reconstruct(self, img):
        a=1

        # mim
        mask = self.test_mask_generator().to("cuda") #!DEBUG
        # mask = torch.stack([data_sample.mask for data_sample in data_samples])
        # img_latent = self.backbone(x, mask)
        img_latent = self.extract_feat(img, mask)
        img_rec = self.recon_neck(img_latent)

        return img_rec


    def extract_feat(self, img, mask=None):
        """Extract features from images."""
        x = self.backbone(img, mask=mask)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward_train(self, img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=False):

        a=1
        losses = dict()

        # mim
        mask = self.mask_generator().to("cuda") #!DEBUG
        # mask = torch.stack([data_sample.mask for data_sample in data_samples])
        # img_latent = self.backbone(x, mask)
        img_latent = self.extract_feat(img, mask)
        img_rec = self.recon_neck(img_latent)
        # img_rec = self.recon_neck(img_latent[0])
        recon_loss = self.recon_head.loss(img_rec, img, mask)
        losses.update({"recon_loss": recon_loss}) #!DEBUG

        a=1

        # segmentation
        # x = self.extract_feat(img, mask)

        if return_feat:
            losses['features'] = img_latent
        loss_decode = self._decode_head_forward_train(img_latent, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        return losses

    def forward_train_orig(self, img, img_metas, gt_semantic_seg, seg_weight=None, return_feat=False):

        # segmentation
        a=1
        x = self.extract_feat(img)

        losses = dict()
        if return_feat:
            losses['features'] = x
        loss_decode = self._decode_head_forward_train(x, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        mask = self.mask_generator().to("cuda") #!DEBUG
        # mask = torch.stack([data_sample.mask for data_sample in data_samples])
        # img_latent = self.backbone(x, mask)
        img_latent = self.extract_feat(img, mask)
        img_rec = self.recon_neck(img_latent)
        # img_rec = self.recon_neck(img_latent[0])
        recon_loss = self.recon_head.loss(img_rec, img, mask)
        losses.update({"recon_loss": recon_loss}) #!DEBUG

        return losses

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        a=1
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        # loss = 2 * log_vars["recon_loss"] + log_vars["decode.loss_seg"]

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars


