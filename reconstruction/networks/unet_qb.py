from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from networks.base_units.swish import CustomSwish
from networks.base_units.ws_conv import WNConv2d
from networks.base_units.quasibinarize import QuasiBinarizingLayer

from networks.unet import UNet

import numpy as np


def get_groups(channels: int) -> int:
    """
    :param channels:
    :return: return a suitable parameter for number of groups in GroupNormalisation'.
    """
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]


class UNet_QB(UNet):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=6,
            padding=True,
            norm="group",
            up_mode='upconv',
            image_size = 128, 
            epsilon=1.0, 
#            latent_sizes_per_pixel=(1,2,4,8,16),
#            latent_sizes_per_pixel=(4,8,16,32,64),
#            latent_sizes_per_pixel=('identity','identity','identity','identity','identity'),
            latent_sizes_per_pixel=(1,2,4,8,16),
            num_top_latent=4096,
            using_heaviside=False
    ):
        """
        QuasiBinarization version of
        
        A modified U-Net implementation [1].

        [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
            Ronneberger et al., 2015 https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
            norm (str): one of 'batch' and 'group'.
                        'batch' will use BatchNormalization.
                        'group' will use GroupNormalization.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        assert(len(latent_sizes_per_pixel) == depth)

        self.image_size = image_size
        self.latent_sizes_per_pixel = latent_sizes_per_pixel

        self.image_average_bias = 1 # nn.Parameter(torch.tensor(np.zeros((1,in_channels,image_size,image_size), dtype=np.float32)))
        self.image_std_bias = 1 # nn.Parameter(torch.tensor(np.ones((1,in_channels,image_size,image_size), dtype=np.float32)))

        self.preneckconvs = nn.ModuleList()
        self.bottlenecks = nn.ModuleList()
        self.postneckconvs = nn.ModuleList()

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            if latent_sizes_per_pixel[i] == 'identity':
                self.preneckconvs.append(
                    nn.Identity()
                )
                self.bottlenecks.append(
                    QuasiBinarizingLayer(
                        2 ** (wf + i) * (image_size//(2**i))**2, 
                        epsilon_per_dimension=epsilon,
                        using_heaviside=using_heaviside
                    )
                )
                self.postneckconvs.append(
                    nn.Identity()
                )
            else:
                self.preneckconvs.append(
                    nn.Conv2d(2 ** (wf + i), latent_sizes_per_pixel[i], kernel_size=1, padding=0)
                )
                self.bottlenecks.append(
                    QuasiBinarizingLayer(
                        latent_sizes_per_pixel[i] * (image_size//(2**i))**2, 
                        epsilon_per_dimension=epsilon,
                        using_heaviside=using_heaviside
                    )
                )
                self.postneckconvs.append(
                    nn.Conv2d(latent_sizes_per_pixel[i], 2 ** (wf + i), kernel_size=1, padding=0)
                )
            prev_channels = 2 ** (wf + i)

        self.top_preneckFC = nn.Linear(prev_channels * (image_size//(2**(depth-1)))**2, num_top_latent)

        if 1:

            self.top_norm1 = nn.GroupNorm(get_groups(num_top_latent), num_top_latent)
            self.top_norm_optional_1 = nn.GroupNorm(get_groups(num_top_latent), num_top_latent)
            self.top_norm_optional_2 = nn.GroupNorm(get_groups(num_top_latent), num_top_latent)

            self.top_FC_optional_1 = nn.Linear(num_top_latent, num_top_latent)
            self.top_FC_optional_2 = nn.Linear(num_top_latent, num_top_latent)

            self.top_preneck_bias = nn.Parameter(torch.tensor(np.zeros((num_top_latent), dtype=np.float32)))
            self.top_postneck_bias = nn.Parameter(torch.tensor(np.zeros((num_top_latent), dtype=np.float32)))

            self.top_norm2 = nn.GroupNorm(get_groups(num_top_latent), num_top_latent)

        # a hack
        if 0:
            self.top_FC_optional_1.weight.data.fill_(0.0)
            self.top_FC_optional_1.bias.data.fill_(0.0)
            self.top_FC_optional_2.weight.data.fill_(0.0)
            self.top_FC_optional_2.bias.data.fill_(0.0)

            self.top_preneckFC.weight.data.fill_(0.0)
            self.top_preneckFC.bias.data.fill_(0.0)

        self.top_bottleneck = QuasiBinarizingLayer(num_top_latent, epsilon_per_dimension=epsilon, using_heaviside=using_heaviside)

        self.top_postneckFC = nn.Linear(num_top_latent, prev_channels * (image_size//(2**(depth-1)))**2)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=3, padding=1)
        self.last_logvar = nn.Conv2d(prev_channels, n_classes, kernel_size=3, padding=1)
        

    def forward_down(self, x):

        blocks = []
        firing_rates = []
        real_firing_rates = []
        unnoised_z = []
        z = []
        for i, down in enumerate(self.down_path):
            x = down(x)

            # make shortcut
            sc = self.preneckconvs[i](x)
            
            sc_shape = sc.shape
            sc = sc.reshape([sc.shape[0], -1])
            res = self.bottlenecks[i](sc)    # bottleneck
            sc = res["x"]

            firing_rates.append(res["expected_firing_rate"])
            real_firing_rates.append(res["real_firing_rate"])
            unnoised_z.append(res["unnoised_x"])
            z.append(res["x"])

            sc = sc.reshape(sc_shape)

            sc = self.postneckconvs[i](sc)

            blocks.append(sc)

            # pooling
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)

        return x, blocks, firing_rates, real_firing_rates, unnoised_z, z

    def forward_up_without_last(self, x, blocks, shortcut_multiplier = 1.0):
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip * shortcut_multiplier)

        return x

    def forward_without_last(self, x, shortcut_multiplier = 1.0):
        x, blocks, firing_rates, real_firing_rates, unnoised_z, z = self.forward_down(x)

        # top bottleneck
        x_shape = x.shape
        x = x.reshape([x.shape[0], -1])
        x = self.top_preneckFC(x)
        
        if 1:
            
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x = self.top_norm1(x)

            x = self.top_norm_optional_1(x)
            x = self.top_FC_optional_1(x)

            x += self.top_preneck_bias

        x = x / 10.0

        res = self.top_bottleneck(x)
        x = res["x"]

        firing_rates.append(res["expected_firing_rate"])
        real_firing_rates.append(res["real_firing_rate"])
        unnoised_z.append(res["unnoised_x"])
        z.append(res["x"])

        if 1:
            x = self.top_norm_optional_2(x)
            x = self.top_FC_optional_2(x)

            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x = self.top_norm2(x)

            x += self.top_postneck_bias



        x = self.top_postneckFC(x)
        x = x.reshape(x_shape)

        x = self.forward_up_without_last(x, blocks, shortcut_multiplier = shortcut_multiplier)
        return x, firing_rates, real_firing_rates, unnoised_z, z

    def forward(self, x, shortcut_multiplier = 1.0):
        # average subtraction
        x = x - self.image_average_bias
        x = x / self.image_std_bias

        # main func
        x, firing_rates, real_firing_rates, unnoised_z, z = self.get_features(x, shortcut_multiplier)

        # reconstruct firing_rates
        firing_rates = torch.stack(firing_rates, dim=1).mean(dim=1)
        real_firing_rates = torch.stack(real_firing_rates, dim=1).mean(dim=1)
        
        # reconstruct unnoised z and z
        batch_size = x.shape[0]
        unnoised_z = [q.view([batch_size, -1]) for q in unnoised_z]
        unnoised_z = torch.concatenate(unnoised_z, dim=1)
        z = [q.view([batch_size, -1]) for q in z]
        z = torch.concatenate(z, dim=1)

        # more accurate
        firing_rates = torch.mean(z, dim=1)
        real_firing_rates = torch.where(unnoised_z > 0.5, 1.0, 0.0).mean(dim=1)

        return {
            'x_hat': self.last(x) * self.image_std_bias + self.image_average_bias, 
            'log_var': self.last_logvar(x), 
            'firing_rate': firing_rates,
            'real_firing_rate': real_firing_rates,
            'unnoised_z': unnoised_z,
            'z': z
        }

    def get_features(self, x, shortcut_multiplier = 1.0):
        return self.forward_without_last(x, shortcut_multiplier)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3):
        super(UNetConvBlock, self).__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


if __name__ == '__main__':
    model = UNet()