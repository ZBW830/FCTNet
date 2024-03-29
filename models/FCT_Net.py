import numpy as np
import torch
import torch.nn as nn

from .ConvNext import convnext_small
from .swin_transformer import SwinTransformer
from torch.nn import functional as F
from timm.models.layers import DropPath, to_3tuple, trunc_normal_

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class GFM_block(nn.Module):
    """
    Do spatial attention to the feature maps of the convolution layer and channel attention to the feature maps of the transformer layer, then do bilinear fusion.
    Args:
        ch_1 (int): Initial number of channels of the spatial feature map, corresponding to the convolutional features.
        ch_2 (int): Initial number of channels in the channel feature map, corresponding to transformer features.
        r_2 (int): Compression ratio of channel feature maps.
        ch_int (int): Number of intermediate layer input channels.
        ch_out (int): Number of output channels.

    """
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(GFM_block, self).__init__()

        # channel attention for F_g, use SE Block
        self.fc1 = nn.Conv2d(ch_2, ch_2 // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // r_2, ch_2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # bi-linear modelling for both
        self.W_g = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_x = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        # self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)
        self.Updim = Conv(ch_int // 2, ch_int, 1, bn=True, relu=True)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.gelu = nn.GELU()


        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, g, x, f):
        W_g = self.W_g(g)
        W_x = self.W_x(x)
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_g, W_x], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_g, W_x], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, X_f], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse) + shortcut
        else:
            return fuse + shortcut


class ConvBNReLU(nn.Module):
    def __init__(self,
                 c_in,
                 c_out,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in,
                              c_out,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x

class conv_block(nn.Module):
    def __init__(self, cin, cout):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_ch1, out_ch, in_ch2=0, attn=False):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = conv_block(in_ch1 + in_ch2, out_ch)

        if attn:
            self.attn_block = RAttention_block(in_ch1, in_ch2, out_ch)
        else:
            self.attn_block = None

    def forward(self, x1, x2=None):

        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            if self.attn_block is not None:
                x2 = self.attn_block(x1, x2)
            x1 = torch.cat([x2, x1], dim=1)
        x = x1
        return self.conv(x)

class Conv2d_batchnorm(torch.nn.Module):
    """
    2D Convolutional layers
    """

    def __init__(
        self,
        num_in_filters,
        num_out_filters,
        kernel_size,
        stride=(1, 1),
        activation="LeakyReLU",
    ):
        """
        Initialization

        Args:
            num_in_filters {int} -- number of input filters
            num_out_filters {int} -- number of output filters
            kernel_size {tuple} -- size of the convolving kernel
            stride {tuple} -- stride of the convolution (default: {(1, 1)})
            activation {str} -- activation function (default: {'LeakyReLU'})
        """
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=num_in_filters,
            out_channels=num_out_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)


    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm(x)

        return self.activation(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class MFCM(torch.nn.Module):
    """
    Implements Multi-scale Feature Compilation Module

    """

    def __init__(self, in_filters1, in_filters2, in_filters3):
        """
        Initialization

        Args:
            in_filters1 (int): number of channels in the first level
            in_filters2 (int): number of channels in the second level
            in_filters3 (int): number of channels in the third level
        """

        super().__init__()

        self.in_filters1 = in_filters1
        self.in_filters2 = in_filters2
        self.in_filters3 = in_filters3
        self.fuse_filters = (
            in_filters1 + in_filters2 + in_filters3
        )

        self.no_param_up = torch.nn.Upsample(scale_factor=2)  # used for upsampling
        self.no_param_down = torch.nn.AvgPool2d(2)  # used for downsampling

        self.ca_1 = ChannelAttention(in_filters1)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(in_filters2)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(in_filters3)
        self.sa_3 = SpatialAttention()

        self.cnv_blks1 = Conv2d_batchnorm(self.fuse_filters, in_filters1, (1, 1))
        self.cnv_blks2 = Conv2d_batchnorm(self.fuse_filters, in_filters2, (1, 1))
        self.cnv_blks3 = Conv2d_batchnorm(self.fuse_filters, in_filters3, (1, 1))

        self.act = torch.nn.LeakyReLU()

    def forward(self, x1, x2, x3):

        x_c1 = x1 + self.cnv_blks1(
                    torch.cat(
                        [
                            x1,
                            self.no_param_up(x2),
                            self.no_param_up(self.no_param_up(x3)),
                        ],
                        dim=1,
                    )
                )

        x_c2 = x2 + self.cnv_blks2(
                    torch.cat(
                        [
                            self.no_param_down(x1),
                            x2,
                            self.no_param_up(x3),
                        ],
                        dim=1,
                    )
                )

        x_c3 = x3 + self.cnv_blks3(
                    torch.cat(
                        [
                            self.no_param_down(self.no_param_down(x1)),
                            self.no_param_down(x2),
                            x3,
                        ],
                        dim=1,
                    )
                )
        x_c1 = self.ca_1(x_c1) * x_c1
        x_c1 = self.sa_1(x_c1) * x_c1

        x_c2 = self.ca_2(x_c2) * x_c2
        x_c2 = self.sa_2(x_c2) * x_c2

        x_c3 = self.ca_3(x_c3) * x_c3
        x_c3 = self.sa_3(x_c3) * x_c3


        return x_c1, x_c2, x_c3

class RAttention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(RAttention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
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
        return x * psi + x

class SegHead(nn.Module):

    def __init__(self, in_ch, mid_dim, num_classes):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = conv_block(in_ch, mid_dim)
        self.conv2 = ConvBNReLU(mid_dim, mid_dim)
        self.final = nn.Conv2d(mid_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.up(x)
        x = self.conv2(x)

        return self.final(x)

class FCTNet(nn.Module):
    def __init__(self, input_size=224, in_chans=3, depths=3, num_classes=2, gt_ds=False, sigmoid=False):
        super(FCTNet, self).__init__()
        self.gt_ds = gt_ds
        self.depths = depths
        self.num_classes = num_classes
        self.sigmoid = sigmoid
        self.cnn_block = convnext_small(in_chans=in_chans)
        self.trans_block = SwinTransformer(img_size=input_size, patch_size=4, in_chans=in_chans,
                                           embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24],
                                           window_size=7, mlp_ratio=4., drop_path_rate=0.2)
        self.fu1 = GFM_block(ch_1=96, ch_2=96, r_2=2, ch_int=96, ch_out=96, drop_rate=0.1)
        self.fu2 = GFM_block(ch_1=192, ch_2=192, r_2=4, ch_int=192, ch_out=192, drop_rate=0.1)
        self.fu3 = GFM_block(ch_1=384, ch_2=384, r_2=8, ch_int=384, ch_out=384, drop_rate=0.1)
        self.fu4 = GFM_block(ch_1=768, ch_2=768, r_2=16, ch_int=768, ch_out=768, drop_rate=0.1)
        self.up1 = Up(in_ch1=768, out_ch=384, in_ch2=384, attn=True)
        self.up2 = Up(in_ch1=384, out_ch=192, in_ch2=192, attn=True)
        self.up3 = Up(in_ch1=192, out_ch=96, in_ch2=96, attn=True)
        self.skip = MFCM(96, 192, 384)
        self.up_final = SegHead(96, 48, num_classes)

        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(384, num_classes, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(192, num_classes, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(96, num_classes, 1))
            print('gt deep supervision was used')
        self.apply(self._init_weights)
        if in_chans==4:
            self.load_weights2()
        else:
            self.load_weights()


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            # nn.init.constant_(m.bias, 0)

    def forward(self, x):
        cnn_features = self.cnn_block(x)
        trans_features = self.trans_block(x)
        fusion_feature1 = self.fu1(cnn_features[0], trans_features[0], None)
        fusion_feature2 = self.fu2(cnn_features[1], trans_features[1], fusion_feature1)
        fusion_feature3 = self.fu3(cnn_features[2], trans_features[2], fusion_feature2)
        fusion_feature4 = self.fu4(cnn_features[-1], trans_features[-1], fusion_feature3)
        cp1, cp2, cp3 = self.skip(fusion_feature1, fusion_feature2, fusion_feature3)
        out1 = self.up1(fusion_feature4, cp3)
        out2 = self.up2(out1, cp2)
        out3 = self.up3(out2, cp1)
        out = self.up_final(out3)
        if self.gt_ds:
            pre1 = self.gt_conv1(out1)
            pre1 = F.interpolate(pre1, scale_factor=16, mode='bilinear', align_corners=True)
            pre2 = self.gt_conv2(out2)
            pre2 = F.interpolate(pre2, scale_factor=8, mode='bilinear', align_corners=True)
            pre3 = self.gt_conv3(out3)
            pre3 = F.interpolate(pre3, scale_factor=4, mode='bilinear', align_corners=True)

            if self.sigmoid:
                return torch.sigmoid(out), torch.sigmoid(pre1), torch.sigmoid(pre2), torch.sigmoid(pre3)
            else:
                return out, pre1, pre2, pre3
        if self.sigmoid:
            return torch.sigmoid(out)
        else:
            return out


    def load_weights(self):
        print('load convnext_small swin_small')
        state_dict1 = torch.load('../weight/swin_small_patch4_window7_224_22k.pth')
        state_dict2 = torch.load('../weight/convnext_small_22k_224.pth')
        msg1 = self.trans_block.load_state_dict(state_dict1['model'], strict=False)
        print(msg1)
        msg2 = self.cnn_block.load_state_dict(state_dict2['model'], strict=False)
        print(msg2)

    def load_weights2(self):
        print('load convnext_small swin_small')
        state_dict1 = torch.load('../weight/swin_small_patch4_window7_224_22k.pth')
        from collections import OrderedDict
        new_state_dict1 = OrderedDict()
        m = ['patch_embed.proj.weight', 'patch_embed.proj.bias']
        for k, v in state_dict1['model'].items():
            if k not in m:
                new_state_dict1[k] = v
        msg1 = self.trans_block.load_state_dict(new_state_dict1, strict=False)
        print(msg1)
        state_dict2 = torch.load('../weight/convnext_small_22k_224.pth')
        new_state_dict2 = OrderedDict()
        m = ['downsample_layers.0.0.weight', 'downsample_layers.0.0.bias']
        for k, v in state_dict2['model'].items():
            if k not in m:
                new_state_dict2[k] = v
        msg2 = self.cnn_block.load_state_dict(new_state_dict2, strict=False)
        print(msg2)

