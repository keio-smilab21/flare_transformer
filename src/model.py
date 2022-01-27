import copy
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor


class FlareTransformer(nn.Module):
    def __init__(self, input_channel, output_channel, sfm_params, mm_params,
                 pretrain_path="None", pretrain_type="None", window=24):
        super(FlareTransformer, self).__init__()
        self.pretrain_type = pretrain_type

        c = copy.deepcopy
        attn = MultiHeadedAttention(mm_params["h"], mm_params["d_model"])
        ff = PositionwiseFeedForward(
            mm_params["d_model"], mm_params["d_ff"], mm_params["dropout"])
        self.trm = Encoder(mm_params["N"], EncoderLayer(
            mm_params["d_model"], c(attn), c(ff), dropout=mm_params["dropout"]))

        self.magnetogram_feature_extractor = ImageFeatureExtractor(
            16, 16, pretrain=False)
        print("pretrain : ", pretrain_path)
        self.magnetogram_feature_extractor.load_state_dict(
            torch.load(pretrain_path), strict=False)
        for param in self.magnetogram_feature_extractor.parameters():
            param.requires_grad = False

        self.feat_model = SunspotFeatureModule(input_channel=input_channel,
                                               output_channel=output_channel,
                                               N=sfm_params["N"],
                                               d_model=sfm_params["d_model"],
                                               h=sfm_params["h"],
                                               d_ff=sfm_params["d_ff"],
                                               dropout=sfm_params["dropout"],
                                               mid_output=2)
        self.generator = nn.Linear(sfm_params["d_model"]+mm_params["d_model"],
                                   output_channel)

        self.linear = nn.Linear(
            window*mm_params["d_model"]*2, sfm_params["d_model"])
        self.softmax = nn.Softmax(dim=1)

        self.generator_feat = nn.Linear(sfm_params["d_model"], output_channel)
        self.generator_image = nn.Linear(256, output_channel)

    def forward(self, img_list, feat):
        for i, img in enumerate(img_list):  # img_list = [bs, k, 256]
            img_output = self.magnetogram_feature_extractor(img)
            if i == 0:
                trm_input = img_output.unsqueeze(0)
            else:
                trm_input = torch.cat(
                    [trm_input, img_output.unsqueeze(0)], dim=0)

        # for window
        img_output = self.trm(trm_input)  # [bs, k, 256]
        # img_output = self.trm(img_output) # [bs, k, 256] # for 2 image trm

        img_output = torch.cat([trm_input, img_output], dim=2)  # id21 res
        img_output = torch.flatten(img_output, 1, 2)  # [bs, k*256]
        img_output = self.linear(img_output)  # [bs, 256]

        feat_output = self.feat_model(feat)
        output = torch.cat((feat_output, img_output), 1)
        output = self.generator(output)
        output = self.softmax(output)

        return output


class SunspotFeatureModule(torch.nn.Module):
    def __init__(self, input_channel=231, output_channel=2, N=6,
                 d_model=256, h=4, d_ff=16, dropout=0.1, mid_output=False):
        super(SunspotFeatureModule, self).__init__()
        self.mid_output = mid_output
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.encoder = Encoder(N, EncoderLayer(
            d_model, c(attn), c(ff), dropout=dropout))

        self.relu = torch.nn.ReLU()
        self.linear_in = torch.nn.Linear(input_channel, d_model)  # 79 -> 200
        self.linear_out = torch.nn.Linear(d_model, input_channel)  # 200 -> 79
        self.bn = torch.nn.BatchNorm1d(input_channel)  # 79
        self.bn2 = torch.nn.BatchNorm1d(d_model)  # 200
        self.generator = torch.nn.Linear(d_model, output_channel)  # 200 -> 2
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        output = self.linear_in(x)
        output = self.bn2(output)
        output = self.relu(output)

        # output = self.linear_mid(output)
        output = output.unsqueeze(1)
        output = self.encoder(output)  # [bs, 1, d_model]
        output = output.squeeze(1)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.linear_out(output)
        output = self.bn(output)
        output = self.relu(output)

        middle_output = x + output
        output = middle_output

        output = self.linear_in(output)
        output = self.bn2(output)
        output = self.relu(output)

        if self.mid_output == 1:
            return output

        # output = self.linear_mid(output)
        output = output.unsqueeze(1)
        output = self.encoder(output)  # [bs, 1, d_model]
        output = output.squeeze(1)
        output = self.bn2(output)
        output = self.relu(output)

        output = self.linear_out(output)
        output = self.bn(output)
        output = self.relu(output)

        output = middle_output + output

        output = self.linear_in(output)
        output = self.bn2(output)
        output = self.relu(output)

        if self.mid_output == 2:
            return output

        output = self.generator(output)
        output = self.softmax(output)

        return output


class Encoder(torch.nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, N, layer):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderLayer(torch.nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


class SublayerConnection(torch.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class LayerNorm(torch.nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class ImageFeatureExtractor(nn.Module):
    def __init__(self, k=16, r=16, pretrain=False):
        super(ImageFeatureExtractor, self).__init__()

        self.pretrain = pretrain
        self.cnn = ConvolutionalLayer(16, 16)
        self.gn = nn.Linear(256, 4)
        self.softmax = nn.Softmax(dim=1)

        if not pretrain:
            print("fine tuning mode")
        else:
            print("pretrain mode")

    def forward(self, img):
        x = self.cnn(img)
        if not self.pretrain:
            return x
        x = self.gn(x)
        x = self.softmax(x)
        return x


class ConvolutionalLayer(nn.Module):
    def __init__(self, k=16, r=16):
        super(ConvolutionalLayer, self).__init__()

        self.input_conv = nn.Conv2d(1, k, kernel_size=2, stride=2)
        self.conv_block1 = ConvolutionalBlock(k=16, r=r)
        self.conv_block2 = ConvolutionalBlock(k=16+2*r, r=r)
        self.conv_block3 = ConvolutionalBlock(k=16+4*r, r=r)
        self.conv_block4 = ConvolutionalBlock(k=16+6*r, r=r)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(4)
        self.linear = nn.Linear(1024, 128)
        self.softmax = nn.Softmax(dim=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=4, stride=2, padding=1)

    def forward(self, img):
        x = img
        mid1 = self.input_conv(x)
        x = self.conv_block1(mid1)  # bs 16
        x = self.maxpool(x)  # [bs, 32, 64, 64]
        x = self.conv(x)  # [bs, 4, 64, 64]
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [bs, 4, 16, 16]
        x = self.conv2(x)  # [bs, 4, 8, 8]
        x = torch.flatten(x, 1, 3)
        return x


class ConvolutionalBlock(nn.Module):
    def __init__(self, k=16, r=16):
        super(ConvolutionalBlock, self).__init__()

        self.conv1 = nn.Conv2d(k, k+r, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(k, k+r, kernel_size=(1, 3),
                               stride=1, padding=(0, 1))
        self.conv3 = nn.Conv2d(k, k+r, kernel_size=(3, 1),
                               stride=1, padding=(1, 0))
        self.conv4 = nn.Conv2d(k, k+r, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(4*(k+r), 4*(k+r), kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(4*(k+r), k+r, kernel_size=3,
                               stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(k+r)
        self.bn2 = nn.BatchNorm2d(4*(k+r))
        self.relu = nn.ReLU()

    def forward(self, img):
        tmp1 = self.conv1(img)
        tmp1 = self.bn1(tmp1)
        tmp1 = self.relu(tmp1)

        tmp2 = self.conv2(img)
        tmp2 = self.bn1(tmp2)
        tmp2 = self.relu(tmp2)

        tmp3 = self.conv3(img)
        tmp3 = self.bn1(tmp3)
        tmp3 = self.relu(tmp3)

        tmp4 = self.conv4(img)
        tmp4 = self.bn1(tmp4)
        tmp4 = self.relu(tmp4)

        x = torch.cat([tmp1, tmp2, tmp3, tmp4], dim=1)  # [bs, 128, 256, 256]

        x = self.conv5(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn1(x)
        x = self.relu(x)

        return x


class CNNModel(nn.Module):
    def __init__(self, output_channel=4, size=2):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        downsample = nn.Sequential(
            conv1x1(16, 8 * 4, 1),
            nn.BatchNorm2d(8 * 4),
        )
        self.layer1 = Bottleneck(
            16, 8, 1, downsample=downsample, norm_layer=nn.BatchNorm2d
        )

        self.layer2 = Bottleneck(
            32, 8, 1, downsample=nn.Sequential(
                conv1x1(32, 8 * 4, 1),
                nn.BatchNorm2d(8 * 4),
            ), norm_layer=nn.BatchNorm2d
        )

        self.avgpool = nn.AdaptiveAvgPool2d((size, size))
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(32*size*size, output_channel)
        self.softmax = nn.Softmax(dim=1)

        self.fc = nn.Linear(32*size*size, 32)
        self.fc2 = nn.Linear(32, output_channel)
        self.bn3 = nn.BatchNorm2d(8 * 4)
        self.dropout = nn.Dropout()

    def forward(self, x):
        # print(x.shape)  # [bs, 1, 512, 512]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)  # [bs, 16, 128, 128]
        x = self.layer1(x)
        # print(x.shape)  # [bs, 32, 128, 128]
        x = self.avgpool(x)
        x = self.flatten(x)
        # print(x.shape)  # [bs, 32*2*2]
        x = self.fc(x)

        x = self.dropout(x)
        x = self.relu(x) # [bs, 32]
        x = self.fc2(x)
        # print(x.shape)  # [bs, 2]
        # sys.exit()
        # sys.exit()
        x = self.softmax(x)

        return x


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)