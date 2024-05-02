# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence, Tuple, Union
from functools import partial

import torch
import torch.nn as nn

from mmengine.model import BaseModel
from mmengine.model.weight_init import trunc_normal_

# from mmpretrain.models import MixTransformer #!DEBUG
# from mmseg.models import MixVisionTransformerCVP
from .mix_transformer_cvp import MixVisionTransformerCVP
from mmseg.models.builder import BACKBONES

# from mmpretrain.registry import MODELS
# from mmpretrain.structures import DataSample
# from .base import BaseSelfSupervisor


@BACKBONES.register_module()
class SimMIMMixVisionTransformer(MixVisionTransformerCVP):
    def __init__(self, **cfg):
        super(SimMIMMixVisionTransformer, self).__init__(**cfg)
        a=1
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims[0]))

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor]) -> Sequence[torch.Tensor]:

        a=1
        if mask is None:
            return super().forward(x)

        else:
            B = x.shape[0]
            outs = []

            if self.handcrafted_tune:
                handcrafted1, handcrafted2, handcrafted3, handcrafted4 = self.prompt_generator.init_handcrafted(x)
            else:
                handcrafted1, handcrafted2, handcrafted3, handcrafted4 = None, None, None, None
            if self.conv_tune:
                a=1
                conv1, conv2, conv3, conv4 = self.prompt_generator.spm(x)
            else:
                conv1, conv2, conv3, conv4 = None, None, None, None

            # stage 1
            x, H, W = self.patch_embed1(x)

            # MIM
            B, L, _ = x.shape
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1. - w) + mask_token * w
            a=1

            if '1' in self.tuning_stage:
                prompt1 = self.prompt_generator.init_prompt(x, handcrafted1, conv1, block_num=1)
                # prompt1 : (handcrated_feature, embedding_feature, conv_feature)
            for i, blk in enumerate(self.block1):
                if '1' in self.tuning_stage:
                    x = self.prompt_generator.get_prompt(x, prompt1, block_num=1, depth_num=i)
                x = blk(x, H, W)
            x = self.norm1(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 2
            x, H, W = self.patch_embed2(x)
            if '2' in self.tuning_stage:
                prompt2 = self.prompt_generator.init_prompt(x, handcrafted2, conv2, block_num=2)
            for i, blk in enumerate(self.block2):
                if '2' in self.tuning_stage:
                    x = self.prompt_generator.get_prompt(x, prompt2, 2, i)
                x = blk(x, H, W)
            x = self.norm2(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 3
            x, H, W = self.patch_embed3(x)
            if '3' in self.tuning_stage:
                prompt3 = self.prompt_generator.init_prompt(x, handcrafted3, conv3, block_num=3)
            for i, blk in enumerate(self.block3):
                if '3' in self.tuning_stage:
                    x = self.prompt_generator.get_prompt(x,prompt3, 3, i)
                x = blk(x, H, W)
            x = self.norm3(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            # stage 4
            x, H, W = self.patch_embed4(x)
            if '4' in self.tuning_stage:
                prompt4 = self.prompt_generator.init_prompt(x, handcrafted4, conv4, block_num=4)
            for i, blk in enumerate(self.block4):
                if '4' in self.tuning_stage:
                    x = self.prompt_generator.get_prompt(x, prompt4, 4, i)
                x = blk(x, H, W)
            x = self.norm4(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            return outs

            # # stage 1
            # # x, hw_shape = self.patch_embed(x)
            # x, H, W = self.patch_embed1(x)
            # B, L, _ = x.shape

            # mask_token = self.mask_token.expand(B, L, -1)
            # w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            # x = x * (1. - w) + mask_token * w

            # if self.use_abs_pos_embed:
            #     x = x + self.absolute_pos_embed

            # x = self.drop_after_pos(x)

            # outs = []
            # for i, stage in enumerate(self.stages):
            #     x, hw_shape = stage(x, hw_shape)
            #     if i in self.out_indices:
            #         norm_layer = getattr(self, f'norm{i}')
            #         out = norm_layer(x)
            #         out = out.view(-1, *hw_shape,
            #                        stage.out_channels).permute(0, 3, 1,
            #                                                    2).contiguous()
            #         outs.append(out)

            # return tuple(outs)


@BACKBONES.register_module()
class mit_b5_cvp_simmim(SimMIMMixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5_cvp_simmim, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1,
            **kwargs)


# @MODELS.register_module()
# class SimMIM(BaseSelfSupervisor): #(BaseModel):
#     """SimMIM.

#     Implementation of `SimMIM: A Simple Framework for Masked Image Modeling
#     <https://arxiv.org/abs/2111.09886>`_.
#     """
#     def __init__(self, **cfg):
#                 #  backbone: dict,
#                 #  neck: Optional[dict] = None,
#                 #  head: Optional[dict] = None,
#                 #  pretrained = None):
#         a=1
#         # super(SimMIM).__init__(cfg)
#     #     self.backbone = backbone
#     #     self.neck = neck
#     #     self.head = head
#     #     init_cfg = dict()
#     #     if pretrained is not None:
#     #         init_cfg = dict(type='Pretrained', checkpoint=pretrained)
#     #     super().__init__(init_cfg=init_cfg)

#     def forward(self,
#                 inputs: Union[torch.Tensor, List[torch.Tensor]],
#                 data_samples: Optional[List[DataSample]] = None, #noneed
#                 mode: str = 'tensor'):
#         """The unified entry for a forward process in both training and test.

#         The method currently accepts two modes: "tensor" and "loss":

#         - "tensor": Forward the backbone network and return the feature
#           tensor(s) tensor without any post-processing, same as a common
#           PyTorch Module.
#         - "loss": Forward and return a dict of losses according to the given
#           inputs and data samples.

#         Args:
#             inputs (torch.Tensor or List[torch.Tensor]): The input tensor with
#                 shape (N, C, ...) in general.
#             data_samples (List[DataSample], optional): The other data of
#                 every samples. It's required for some algorithms
#                 if ``mode="loss"``. Defaults to None.
#             mode (str): Return what kind of value. Defaults to 'tensor'.

#         Returns:
#             The return type depends on ``mode``.

#             - If ``mode="tensor"``, return a tensor or a tuple of tensor.
#             - If ``mode="loss"``, return a dict of tensor.
#         """
#         if mode == 'tensor':
#             feats = self.extract_feat(inputs)
#             return feats
#         elif mode == 'loss':
#             return self.loss(inputs, data_samples)
#         else:
#             raise RuntimeError(f'Invalid mode "{mode}".')

#     def extract_feat(self, inputs: torch.Tensor):
#         return self.backbone(inputs, mask=None)

#     def loss(self, inputs: torch.Tensor, data_samples: List[DataSample],
#              **kwargs) -> Dict[str, torch.Tensor]:
#         """The forward function in training.

#         Args:
#             inputs (List[torch.Tensor]): The input images.
#             data_samples (List[DataSample]): All elements required
#                 during the forward function.

#         Returns:
#             Dict[str, torch.Tensor]: A dictionary of loss components.
#         """
#         mask = torch.stack([data_sample.mask for data_sample in data_samples])

#         img_latent = self.backbone(inputs, mask)
#         img_rec = self.neck(img_latent[0])
#         loss = self.head.loss(img_rec, inputs, mask)
#         losses = dict(loss=loss)

#         return losses

