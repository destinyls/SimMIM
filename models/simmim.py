# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------
import copy
import random

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from timm.models.layers import trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer

from .deconv_up import DeconvUp

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class SwinTransformerForSimMIM(SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0., std=.02)
        self.neck = DeconvUp(input_channels=self.num_features)
        self.projector = MLP(dim=self.neck.out_channels, projection_size=256)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, mask=None):
        x = self.patch_embed(x)
        B, L, _ = x.shape

        if mask is not None:
            mask_tokens = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
            x = x * (1. - w) + mask_tokens * w

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        x = x.transpose(1, 2)
        B, C, L = x.shape
        H = W = int(L ** 0.5)
        x = x.reshape(B, C, H, W)

        # stop gradient propagation
        z = self.neck(x)
        z = self.avg_pool(z)
        z = z.view(z.size(0), -1)
        out_projector = self.projector(z)

        return x, out_projector

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {'mask_token'}


class VisionTransformerForSimMIM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        assert mask is not None
        
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        '''
        neck = DeconvUp(input_channels=encoder.num_features)
        projector = self._build_mlp(num_layers=3, input_dim=neck.out_channels, mlp_dim=4096, output_dim=256)
        '''
        self.encoder = encoder
        self.momentum_encoder = copy.deepcopy(self.encoder)
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.encoder_stride = encoder_stride
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.use_momentum = True
        self.predictor = MLP(dim=256, projection_size=256)
        
    def forward(self, x, mask, epoch, m=0.99):
        global_crop_t, global_crop_s, local_crops = x[0], x[1], x[2:]

        # mim loss
        z, out_sg = self.encoder(global_crop_s, mask)
        out_sg = self.predictor(out_sg)
        x_rec = self.decoder(z)
        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(global_crop_s, x_rec, reduction='none')
        mim_loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        # contrast loss
        student_outputs = []
        for i in range(len(local_crops)):
            _, out_sl = self.encoder(local_crops[i])
            out_sl = self.predictor(out_sl)
            student_outputs.append(out_sl)
        student_outputs.append(out_sg)

        teacher_outputs = []
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            _, out_tg = self.momentum_encoder(global_crop_t)
            teacher_outputs.append(out_tg.detach())

        cl_loss, n_cl_loss = 0.0, 0
        for i in range(len(student_outputs)):
            cl_loss += self.loss_fn(teacher_outputs[0], student_outputs[i]).mean()
            n_cl_loss += 1
        cl_loss = cl_loss / n_cl_loss

        return mim_loss, cl_loss


    
    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}
    
    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def loss_fn(self, x, y):
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

def build_simmim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        encoder = SwinTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM)
        encoder_stride = 32
    elif model_type == 'vit':
        encoder = VisionTransformerForSimMIM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = SimMIM(encoder=encoder, encoder_stride=encoder_stride)

    return model
