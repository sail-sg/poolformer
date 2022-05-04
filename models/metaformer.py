# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MetaFormer implementation with hybrid stages
"""
from typing import Sequence
from functools import partial, reduce
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


from .poolformer import PatchEmbed, LayerNormChannel, GroupNorm, Mlp



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }

class AddPositionEmb(nn.Module):
    """Module to add position embedding to input features
    """
    def __init__(
        self, dim=384, spatial_shape=[14, 14],
        ):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must by a sequence or int, ' \
            f'get {type(spatial_shape)} instead.'
        if len(spatial_shape) == 1:
            embed_shape = list(spatial_shape) + [dim]
        else:
            embed_shape = [dim] + list(spatial_shape)
        self.pos_embed = nn.Parameter(torch.zeros(1, *embed_shape))
    def forward(self, x):
        return x+self.pos_embed


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Attention(nn.Module):
    """Attention module that can take tensor with [B, N, C] or [B, C, H, W] as input.
    Modified from: 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % head_dim == 0, 'dim should be divisible by head_dim'
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        shape = x.shape
        if len(shape) == 4:
            B, C, H, W = shape
            N = H * W
            x = torch.flatten(x, start_dim=2).transpose(-2, -1) # (B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # trick here to make q@k.t more stable
        attn = (q * self.scale) @ k.transpose(-2, -1)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose(-2, -1).reshape(B, C, H, W)

        return x


class SpatialFc(nn.Module):
    """SpatialFc module that take features with shape of (B,C,*) as input.
    """
    def __init__(
        self, spatial_shape=[14, 14], **kwargs, 
        ):
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = [spatial_shape]
        assert isinstance(spatial_shape, Sequence), \
            f'"spatial_shape" must by a sequence or int, ' \
            f'get {type(spatial_shape)} instead.'
        N = reduce(lambda x, y: x * y, spatial_shape)
        self.fc = nn.Linear(N, N, bias=False)

    def forward(self, x):
        # input shape like [B, C, H, W]
        shape = x.shape
        x = torch.flatten(x, start_dim=2) # [B, C, H*W]
        x = self.fc(x) # [B, C, H*W]
        x = x.reshape(*shape) # [B, C, H, W]
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    --dim: embedding dim
    --token_mixer: token mixer module
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, 
                 token_mixer=nn.Identity, 
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, token_mixer=nn.Identity, 
                 mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=LayerNormChannel, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(MetaFormerBlock(
            dim, token_mixer=token_mixer, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class MetaFormer(nn.Module):
    """
    MetaFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios: the embedding dims and mlp ratios for the 4 stages
    --token_mixers: token mixers of different stages
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --add_pos_embs: position embedding modules of different stages
    """
    def __init__(self, layers, embed_dims=None, 
                 token_mixers=None, mlp_ratios=None, 
                 norm_layer=LayerNormChannel, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 downsamples=None, down_patch_size=3, down_stride=2, down_pad=1, 
                 add_pos_embs=None, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 **kwargs):

        super().__init__()


        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])
        if add_pos_embs is None:
            add_pos_embs = [None] * len(layers)
        if token_mixers is None:
            token_mixers = [nn.Identity] * len(layers)
        # set the main block in network
        network = []
        for i in range(len(layers)):
            if add_pos_embs[i] is not None:
                network.append(add_pos_embs[i](embed_dims[i]))
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 token_mixer=token_mixers[i], mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()

        self.apply(self.cls_init_weights)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        x = self.norm(x)
        # for image classification
        cls_out = self.head(x.mean([-2, -1]))
        return cls_out

model_urls = {
    "metaformer_id_s12": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_id_s12.pth.tar",
    "metaformer_pppa_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_pppa_s12_224.pth.tar",
    "metaformer_ppaa_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_ppaa_s12_224.pth.tar",
    "metaformer_pppf_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_pppf_s12_224.pth.tar",
    "metaformer_ppff_s12_224": "https://github.com/sail-sg/poolformer/releases/download/v1.0/metaformer_ppff_s12_224.pth.tar",
}


@register_model
def metaformer_id_s12(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    token_mixers = [nn.Identity] * len(layers)
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        url = model_urls['metaformer_id_s12']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def metaformer_pppa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, None,
        partial(AddPositionEmb, spatial_shape=[7, 7])]
    token_mixers = [Pooling, Pooling, Pooling, Attention]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        add_pos_embs=add_pos_embs,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['metaformer_pppa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def metaformer_ppaa_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    add_pos_embs = [None, None, 
        partial(AddPositionEmb, spatial_shape=[14, 14]), None]
    token_mixers = [Pooling, Pooling, Attention, Attention]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        add_pos_embs=add_pos_embs,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['metaformer_ppaa_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def metaformer_pppf_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    token_mixers = [Pooling, Pooling, Pooling,
        partial(SpatialFc, spatial_shape=[7, 7]),
        ]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        url = model_urls['metaformer_pppf_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def metaformer_ppff_s12_224(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    token_mixers = [Pooling, Pooling, 
        partial(SpatialFc, spatial_shape=[14, 14]), 
        partial(SpatialFc, spatial_shape=[7, 7]),
        ]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = MetaFormer(
        layers, embed_dims=embed_dims,
        token_mixers=token_mixers,
        mlp_ratios=mlp_ratios,
        norm_layer=GroupNorm,
        downsamples=downsamples,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        url = model_urls['metaformer_ppff_s12_224']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model





