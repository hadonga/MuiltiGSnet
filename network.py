"""
Transformer from ViT. Depth-wise Separable Unet from SmaAt-Unet
Licensed under MIT License [see LICENSE].
This work is inspired from their public code.
"""

''' to delete
2021.1.18
各种网络结构，用于model里面调用
2021.2.15
加入 DSUnet
2021.4.2
加入 Trans DSUnet
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging
import math

import ml_collections
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

our_in_ch=64
our_out_ch=3

# ---------------------------------------------------------------------------- #
# Trans DSUnet 网络   1362.053899M
# ---------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}

def get_net_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    # config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 384# 768 #3072  # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072 #12288  # 3072
    config.transformer.num_heads = 6
    config.transformer.num_layers = 6
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    # config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 3
    config.activation = 'softmax'
    return config

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x):  #（B，64，768）
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x #（B，64，768）

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size=8, in_channels=512):  # img_size= 8  in_channels=512
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)  #(8,8)
        patch_size = (1, 1)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])  # (8*8=64)
        self.patch_embeddings = Conv2d(in_channels=in_channels,  # 512   512* 3072 这部分的参数量较多
                                       out_channels=config.hidden_size,  #768 # 3072
                                       kernel_size=patch_size,  # (1,1)
                                       stride=patch_size)  # (1,1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))  # (24,64,768)
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):  # (B,512,8,8)
        x = self.patch_embeddings(x)  # in:(B,512,8,8)  out:(B, 768,8,8)  使用一次 cnn
        print("checkpoint - x1 size: ", x.size())
        x = x.flatten(2)  # (B,768,64)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)   # (B,64,768)
        print("checkpoint - x2 size: ", x.size())
        b= self.position_embeddings
        print("checkpoint - b size: ", b.size())
        embeddings = x + self.position_embeddings  # (B,64,768)  为为啥要加只是zeros的一个矩阵？ 并没有位置信息。
        embeddings = self.dropout(embeddings)
        return embeddings  #

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()  # 生成空list并注册到网络
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)  # 3072
        self.conv = nn.Conv2d(in_channels=768, out_channels = 256, kernel_size = 1)  # 3072->256 ,但是HW的参数没对齐？
        self.up=nn.UpsamplingBilinear2d(scale_factor=2)
        for _ in range(config.transformer["num_layers"]):  # 12 transformer的层
            layer = Block(config) ##（B，64，768）
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):  # hidden_states= (B,64,768)
        for layer_block in self.layer:  # list[12] 12条路线
            hidden_states = layer_block(hidden_states)  # （B，64，768）放入 12条线路处理，输出为：#（B，64，768）
        encoded = self.encoder_norm(hidden_states)
        B, n_patch, hidden = encoded.size()  # (B,64,768)  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))  # n_patch=64
        x = encoded.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w) # (B,768,8,8)
        x = self.conv(x) #（B，768，8，8） ??? 需要的输出是 B，512，16，16 论文中使用了 upsampling
        x = self.up(x) #(B,512,16,16)
        return x


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)  # img_size=8
        self.encoder = Encoder(config)

    def forward(self, input_ids):  # 入口: (B,512,8,8)
        embedding_output = self.embeddings(input_ids)  #  embedding_output= (B, 64, 768)
        encoded = self.encoder(embedding_output)  #  output= (B,256,8,8)
        return encoded

# DSC Module
class OutConv(nn.Module):  # 一般的1x1conv2d.
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):  # DSC
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size,
                                   padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConvDS(nn.Module):  # 2次DSC
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer,
                                   padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer,
                                   padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownDS(nn.Module):  # e.g
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # stride=2
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):  #
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]  # H2-H1
        diffX = x2.size()[3] - x1.size()[3]  # W2-W1

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])  # x1 and x2  same size
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):  # Conv2d
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# CBAM Module
class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

# Trans + DSC + CBAM
class TransDSUnet_lite(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(TransDSUnet_lite, self).__init__()
        trans_config = get_net_config()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.n_channels, 64,
                                kernels_per_layer=kernels_per_layer)  # (64,128,128) -> (64,128,128)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)  # (64,128,128)-> (128,64,64)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)  # (128,64,64)-> (256,32,32)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)  # (256,32,32)-> (512,16,16)
        self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio) #(512,16,16)

        factor = 2 if self.bilinear else 1
        self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)  # (512,16,16)-> (512,8,8)
        self.trans = Transformer(config=trans_config, img_size=8)  # out:(512,16,16)

        self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer) # (1024,8,8) -> (256,16,16)
        self.up2 = UpDS(512, 256 // factor, self.bilinear,
                        kernels_per_layer=kernels_per_layer)  # (512,16,16) -> (128,32,32)
        self.up3 = UpDS(256, 128 // factor, self.bilinear,
                        kernels_per_layer=kernels_per_layer)  # (256,32,32) -> (128,64,64)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)  # (128,64,64) -> (64,128,128)
        self.outc = OutConv(64, self.n_classes)  # (64,128,128)->(3,128,128)

    def forward(self, x):
        x1 = self.inc(x) # x1: (64,128,128)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1) # x2 : (128,64,64)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)  # x3 : (256,32,32)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)  # x4 : (512,16,16)
        x4Att = self.cbam3(x4)
        x5 = self.down3(x4)  # x5：（512，8，8）
        x5Trans = self.cbam5(x5) # trans:(512,16,16)

        x = self.up1(x5Trans, x4Att)
        x = self.up2(x, x3Att)
        x = self.up3(x, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


# class TransDSUnet(nn.Module):
#     def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
#         super(TransDSUnet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         kernels_per_layer = kernels_per_layer
#         self.bilinear = bilinear
#         reduction_ratio = reduction_ratio
#
#         self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
#         self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
#         self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
#         self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
#         self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
#         self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
#         self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
#         self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
#         factor = 2 if self.bilinear else 1
#         self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
#         self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
#         self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
#         self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
#         self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
#         self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
#
#         self.outc = OutConv(64, self.n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x1Att = self.cbam1(x1)
#         x2 = self.down1(x1)
#         x2Att = self.cbam2(x2)
#         x3 = self.down2(x2)
#         x3Att = self.cbam3(x3)
#         x4 = self.down3(x3)
#         x4Att = self.cbam4(x4)
#         x5 = self.down4(x4)
#         x5Att = self.cbam5(x5)
#         x = self.up1(x5Att, x4Att)
#         x = self.up2(x, x3Att)
#         x = self.up3(x, x2Att)
#         x = self.up4(x, x1Att)
#         logits = self.outc(x)
#         return logits


# ---------------------------------------------------------------------------- #
# DSUnet  0.27M paramerters
# --------------- ------------------------------------------------------------- #
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernels_per_layer, kernel_size=kernel_size, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConvDS(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None, kernels_per_layer=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, mid_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, kernels_per_layer=kernels_per_layer, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownDS(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernels_per_layer=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpDS(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernels_per_layer=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvDS(in_channels, out_channels, in_channels // 2, kernels_per_layer=kernels_per_layer)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvDS(in_channels, out_channels, kernels_per_layer=kernels_per_layer)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels)
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

# class SmaAt_UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
#         super(SmaAt_UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         kernels_per_layer = kernels_per_layer
#         self.bilinear = bilinear
#         reduction_ratio = reduction_ratio
#
#         self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
#         self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
#         self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
#         self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
#         self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
#         self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
#         self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
#         self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
#         factor = 2 if self.bilinear else 1
#         self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
#         self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
#         self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
#         self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
#         self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
#         self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
#
#         self.outc = OutConv(64, self.n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x1Att = self.cbam1(x1)
#         x2 = self.down1(x1)
#         x2Att = self.cbam2(x2)
#         x3 = self.down2(x2)
#         x3Att = self.cbam3(x3)
#         x4 = self.down3(x3)
#         x4Att = self.cbam4(x4)
#         x5 = self.down4(x4)
#         x5Att = self.cbam5(x5)
#         x = self.up1(x5Att, x4Att)
#         x = self.up2(x, x3Att)
#         x = self.up3(x, x2Att)
#         x = self.up4(x, x1Att)
#         logits = self.outc(x)
#         return logits

class DS_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(DS_Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        reduction_ratio = reduction_ratio

        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down2 = DownDS(128, 256 // factor, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256 // factor, reduction_ratio=reduction_ratio)
        # self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
        # self.cbam4 = CBAM(512, reduction_ratio=reduction_ratio)
        # factor = 2 if self.bilinear else 1
        # self.down4 = DownDS(512, 1024 // factor, kernels_per_layer=kernels_per_layer)
        # self.cbam5 = CBAM(1024 // factor, reduction_ratio=reduction_ratio)
        # self.up1 = UpDS(1024, 512 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        # self.up2 = UpDS(512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up4 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)

        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        # x4 = self.down3(x3)
        # x4Att = self.cbam4(x4)
        # x5 = self.down4(x4)
        # x5Att = self.cbam5(x5)
        # x = self.up1(x5Att, x4Att)
        # x = self.up2(x3Att, x2Att)
        x = self.up3(x3Att, x2Att)
        x = self.up4(x, x1Att)
        logits = self.outc(x)
        return logits


# ---------------------------------------------------------------------------- #
# Original MLP
# ---------------------------------------------------------------------------- #
class Org_Mlp(nn.Module):
    def __init__(self,num_input_features=64):
        super().__init__()
        print("Using MLP...")

        self.cls_mlp= nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.nor_mlp = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.cls_fc = nn.Linear(64,3)
        self.nor_fc = nn.Linear(64,3)

    def forward(self, features):
        cls_features = self.cls_mlp(features)
        cls_preds = self.cls_fc(cls_features.permute(0,2,3,1))

        nor_features = self.nor_mlp(features)
        nor_preds = self.nor_fc(nor_features.permute(0,2,3,1))
        return cls_preds.permute(0,3,1,2) #,nor_preds.permute(0,3,1,2)

# ---------------------------------------------------------------------------- #
# New_net  MLP&unet
# ---------------------------------------------------------------------------- #

class New_net(nn.Module):
    def __init__(self,in_ch=our_in_ch, out_ch=our_out_ch, num_input_features=64):
        super(New_net,self).__init__()
        print("Using New network...")

        self.cls_mlp= nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        )

        filters = [64, 128, 256, 512, 1024]
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])
        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])
        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])
        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        self.Conv = nn.Conv2d(filters[0], 32, kernel_size=1, stride=1, padding=0)

        self.cls_fc1 = nn.Linear(64,32)
        self.cls_fc2 = nn.Linear(64,3)

    def forward(self, features):
        e1 = self.Conv1(features)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        unet_features = self.Conv(d2)
        mlp_features = self.cls_mlp(features)
        mlp_features= self.cls_fc1(mlp_features.permute(0,2,3,1))
        mlp_features= mlp_features.permute(0,3,1,2)
        com_features = torch.cat((unet_features,mlp_features),1)
        cls_preds= self.cls_mlp(com_features)
        cls_preds = self.cls_fc2(cls_preds.permute(0, 2, 3, 1))

        return  cls_preds.permute(0,3,1,2)


# ---------------------------------------------------------------------------- #
# Unet
# ---------------------------------------------------------------------------- #
class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=our_in_ch, out_ch=our_out_ch):
        super(U_Net, self).__init__()
        print("Using U_Net...")
        # in_ch =64, out_ch= ?
        filters = [64, 128, 256, 512, 1024]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out

# 未使用R2U_Net
class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        # out = self.active(out)

        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=our_in_ch, output_ch=our_out_ch):
        super(AttU_Net, self).__init__()
        print("Using AttU_Net...")

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


#未使用R2AttU_Net
class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, in_ch=our_in_ch, out_ch=our_out_ch, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out

'''
unet++ network forked from https://github.com/ZJUGiveLab/UNet-Version/blob/master/
'''
class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output

class NestedUNet(nn.Module):  # unet2plus
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, in_channels=our_in_ch, n_classes=our_out_ch):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output

