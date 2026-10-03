from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from networks.base_units.swish import CustomSwish
from networks.base_units.ws_conv import WNConv2d
from networks.base_units.quasibinarize import QuasiBinarizingLayer
from networks.base_units.dcae_residual_autoencoding import ResidualDown, ResidualUp
from networks.base_units.attn_block import attn_block, SpatialAttn, FFN

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


def make_norm2d(channels: int, norm_type: str = "group") -> nn.Module:
    """Per-sample GroupNorm (default) or legacy BatchNorm2d."""
    if norm_type == "group":
        return nn.GroupNorm(get_groups(channels), channels)
    elif norm_type == "batch":
        return nn.BatchNorm2d(channels)
    raise ValueError(norm_type)



class ParamAtan(nn.Module):
    """
    Parametrized arctangent activation:
    y = alpha * atan(beta * x + gamma) + delta
    All parameters are trainable.
    """
    def __init__(self, init_alpha=1.0, init_beta=1.0, init_gamma=0.0, init_delta=0.0):
        super().__init__()
        # Trainable parameters
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32))

    def forward(self, x):
        return self.alpha * torch.atan(self.beta * x + self.gamma) + self.delta



class UNet_QB(UNet):
    def __init__(
            self,
            in_channels=1,
            n_classes=2,
            depth=5,
            wf=4,
            padding=True,
            norm="group",
#            up_mode='upconv',
            up_mode='residual_autoencoder',
            image_size = 128, 
            epsilon=1.0, 
#            latent_sizes_per_pixel=(0,0,0,0,0),
            latent_sizes_per_pixel=(1,2,4,8,16),
#            latent_sizes_per_pixel=(4,8,16,32,64),
#            latent_sizes_per_pixel=('identity','identity','identity','identity','identity'),
            num_top_latent=16384, # 4096,
            mid_channels_per_pixel = 32,
            using_heaviside=False,
            adding_noise_in_test=False,
            using_identity_connection=True,
            attention_gate=True,
            top_mixer="attn",        # "attn" (token transformer, v31 default) or "fc" (legacy)
            top_attn_depth=2,        # number of (SpatialAttn + FFN) blocks before and after the bottom QB
            norm_type="group",       # "group" (per-sample, v31 default) or "batch" (legacy)
            top_pos="abs",           # v32: "abs" = learned absolute 2D positional embedding on the 8x8 tokens, "none" = v31
            max_channels=0,          # v32: cap on the channel width 2**(wf+i) of every level (0 = no cap)
            top_mid_channels=None    # v32: channels per pixel entering the FC top mixer (None = mid_channels_per_pixel)
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
        assert up_mode in ('upconv', 'upsample', 'residual_autoencoder')
        if up_mode == 'residual_autoencoder':
            self.down_mode = 'residual_autoencoder'
        else:
            self.down_mode = None
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        assert(len(latent_sizes_per_pixel) == depth)

        self.image_size = image_size
        self.latent_sizes_per_pixel = latent_sizes_per_pixel
        self.top_mixer = top_mixer
        self.norm_type = norm_type
        assert top_mixer in ("attn", "fc")
        assert top_pos in ("none", "abs")
        self.top_pos = top_pos
        # v32: channel width of level i (capped for deep variants, e.g. depth 7 with max_channels 256)
        ch = lambda i: min(2 ** (wf + i), max_channels) if max_channels else 2 ** (wf + i)
        self.level_channels = [ch(i) for i in range(depth)]
        if top_mid_channels:
            mid_channels_per_pixel = top_mid_channels

        self._using_heaviside = using_heaviside
        self._adding_noise_in_test = adding_noise_in_test

        self.image_average_bias = 1 # nn.Parameter(torch.tensor(np.zeros((1,in_channels,image_size,image_size), dtype=np.float32)))
        self.image_std_bias = 1 # nn.Parameter(torch.tensor(np.ones((1,in_channels,image_size,image_size), dtype=np.float32)))

        # down blocks

        self.preneckconvs = nn.ModuleList()
        self.bottlenecks = nn.ModuleList()
        self.postneckconvs = nn.ModuleList()

        self.down_path = nn.ModuleList()
        if self.down_mode == 'residual_autoencoder':
            self.residual_down = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, ch(i), padding, norm=norm, using_identity=using_identity_connection)
            )
            if self.down_mode == 'residual_autoencoder' and i < depth - 1:
                # (no downsampling after the deepest level; v30 created an unused module here)
                self.residual_down.append(ResidualDown(ch(i), ch(i)))

            if i == depth - 1:
                # v31: the deepest skip was computed (and counted in z) but never used by the decoder -> removed
                self.preneckconvs.append(None)
                self.bottlenecks.append(None)
                self.postneckconvs.append(None)
            elif latent_sizes_per_pixel[i] == 'identity':
                self.preneckconvs.append(
                    nn.Identity()
                )
                self.bottlenecks.append(
                    QuasiBinarizingLayer(
                        ch(i) * (image_size//(2**i))**2, 
                        epsilon_per_dimension=epsilon,
                        using_heaviside=using_heaviside,
                        adding_noise_in_test=adding_noise_in_test
                    )
                )
                self.postneckconvs.append(
                    nn.Identity()
                )
            else:
                if 0<latent_sizes_per_pixel[i]:
                    self.preneckconvs.append(
                        nn.Conv2d(ch(i), latent_sizes_per_pixel[i], kernel_size=1, padding=0)
                    )
                    self.bottlenecks.append(
                        QuasiBinarizingLayer(
                            latent_sizes_per_pixel[i] * (image_size//(2**i))**2, 
                            epsilon_per_dimension=epsilon,
                            using_heaviside=using_heaviside,
                            adding_noise_in_test=adding_noise_in_test
                        )
                    )
                    self.postneckconvs.append(
                        nn.Conv2d(latent_sizes_per_pixel[i], ch(i), kernel_size=1, padding=0)
                    )
                else:
                    self.preneckconvs.append(None)
                    self.bottlenecks.append(None)
                    self.postneckconvs.append(None)
            prev_channels = ch(i)

        # top block

        top_input_image_size = (image_size//(2**(depth-1)))
        assert num_top_latent % (top_input_image_size**2) == 0
        top_latent_ch = num_top_latent // (top_input_image_size**2)

        if top_mixer == "attn":
            # v31: token transformer on the 8x8 grid replaces the two dense FC layers (~134M params)
            self.top_pre_mixer = nn.Sequential(*[nn.Sequential(SpatialAttn(prev_channels), FFN(prev_channels))
                                                 for _ in range(top_attn_depth)])
            self.top_pre_proj = nn.Conv2d(prev_channels, top_latent_ch, kernel_size=1, padding=0)
            self.top_pre_norm = make_norm2d(top_latent_ch, norm_type)
            self.top_post_proj = nn.Conv2d(top_latent_ch, prev_channels, kernel_size=1, padding=0)
            self.top_post_mixer = nn.Sequential(*[nn.Sequential(SpatialAttn(prev_channels), FFN(prev_channels))
                                                  for _ in range(top_attn_depth)])
            if top_pos == "abs":
                # v32: the attention mixer is permutation-equivariant (no position information except at the
                # zero-padded border of the depth-wise conv in FFN), whereas the v30 FC layers had
                # position-specific weights. A learned absolute embedding per 8x8 token restores this.
                # Separate tables before and after the QB layer; the post table is added after the QB, so it
                # carries no information about the input and does not affect the information budget.
                self.top_pos_pre = nn.Parameter(torch.zeros(1, prev_channels, top_input_image_size, top_input_image_size))
                self.top_pos_post = nn.Parameter(torch.zeros(1, prev_channels, top_input_image_size, top_input_image_size))
                nn.init.trunc_normal_(self.top_pos_pre, std=0.02)
                nn.init.trunc_normal_(self.top_pos_post, std=0.02)
        else:
            # legacy (v30) dense bottleneck
            self.top_prepreneckConv = nn.Conv2d(prev_channels,mid_channels_per_pixel, kernel_size=1, padding=0)
            self.top_norm_interpre = make_norm2d(mid_channels_per_pixel, norm_type)
            self.top_preneckFC = nn.Linear(mid_channels_per_pixel * top_input_image_size**2, num_top_latent)
            self.top_pre_shortcut_conv = nn.Conv2d(prev_channels, top_latent_ch, kernel_size=1, padding=0)
            # per-sample normalisation over the whole latent vector (GroupNorm with 1 group) or legacy BN1d
            self.top_pre_batchnorm = nn.GroupNorm(1, num_top_latent) if norm_type == "group" else nn.BatchNorm1d(num_top_latent)
            self.top_postneckFC = nn.Linear(num_top_latent, mid_channels_per_pixel * top_input_image_size**2)
            self.top_norm_interpost = make_norm2d(mid_channels_per_pixel, norm_type)
            self.top_postpostneckConv = nn.Conv2d(mid_channels_per_pixel, prev_channels, kernel_size=1, padding=0)
            self.top_post_shortcut_conv = nn.Conv2d(top_latent_ch, prev_channels, kernel_size=1, padding=0)

        self.top_bottleneck = QuasiBinarizingLayer(num_top_latent, epsilon_per_dimension=epsilon, using_heaviside=using_heaviside, adding_noise_in_test=adding_noise_in_test)

        self.patan = ParamAtan(init_beta=0.01)

        # up blocks

        self.up_path = nn.ModuleList()
        current_image_size = self.image_size // (2 ** (depth-1))
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, ch(i), up_mode, padding, norm=norm, islast=(i==0), attention_gate=attention_gate,  attention_selector=True, image_size=current_image_size, using_identity=using_identity_connection, norm_type=norm_type)
            )
            prev_channels = ch(i)
            current_image_size = current_image_size * 2

        # last convs

        self.last = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(prev_channels, n_classes, kernel_size=3, padding=1),
        )
        self.last_logvar = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(prev_channels, prev_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(prev_channels, n_classes, kernel_size=3, padding=1),
        )

    #dynamic setter / getter (test-time behaviour selecter)

    @property
    def using_heaviside(self):
        return self._using_heaviside
    
    @using_heaviside.setter
    def using_heaviside(self, value):
        assert isinstance(value, bool)
        self._using_heaviside = value
        for bottleneck in self.bottlenecks:
            if bottleneck is not None:
                bottleneck.using_heaviside = value
        self.top_bottleneck.using_heaviside = value

    @property
    def adding_noise_in_test(self):
        return self._adding_noise_in_test
    
    @adding_noise_in_test.setter
    def adding_noise_in_test(self, value):
        assert isinstance(value, bool)
        self._adding_noise_in_test = value
        for bottleneck in self.bottlenecks:
            if bottleneck is not None:
                bottleneck.adding_noise_in_test = value
        self.top_bottleneck.adding_noise_in_test = value

    def forward_down(self, x):

        blocks = []
        firing_rates = []
        real_firing_rates = []
        unnoised_z = []
        z = []
        for i, down in enumerate(self.down_path):
            x = down(x)

            # make U-net skip connection with quasibinary bottleneck
            if self.preneckconvs[i] is None:
                sc = None
            else:
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
                if self.down_mode == 'residual_autoencoder':
                    x = self.residual_down[i](x)
                else:
                    x = F.avg_pool2d(x, 2)

        return x, blocks, firing_rates, real_firing_rates, unnoised_z, z

    def forward_up_without_last(self, x, blocks, shortcut_multiplier = 1.0):
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            if skip is None:
                x = up(x, None)
            else:
                x = up(x, skip * shortcut_multiplier)

        return x

    def forward_without_last(self, x, shortcut_multiplier = 1.0):

        # down paths
        
        x, blocks, firing_rates, real_firing_rates, unnoised_z, z = self.forward_down(x)

        # top bottleneck

        batch_size = x.shape[0]

        if self.top_mixer == "attn":
            if self.top_pos == "abs":
                x = x + self.top_pos_pre
            h = self.top_pre_mixer(x)
            h = self.top_pre_proj(h)
            h = self.top_pre_norm(h)
            h_shape = h.shape
            h = h.reshape([batch_size, -1])
            h = self.patan(h)  # to avoid initial vanishing gradient due to sigmoid
            res = self.top_bottleneck(h)
            z_top = res["x"]
            top_recon_loss = torch.zeros((batch_size, 1, 1, 1), device=x.device, dtype=x.dtype)
            h = z_top.reshape(h_shape)
            h = self.top_post_proj(h)
            if self.top_pos == "abs":
                h = h + self.top_pos_post
            x = self.top_post_mixer(h)
        else:
            shortcut = x
            x = self.top_prepreneckConv(x)
            x = self.top_norm_interpre(x)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x_shape = x.shape
            x = x.reshape([batch_size, -1])
            x = self.top_preneckFC(x)
            x = self.top_pre_batchnorm(x)
            x = x + self.top_pre_shortcut_conv(shortcut).reshape(x.shape)
            x_preneck = x
            x = self.patan(x)  # to avoid initial vanishing gradient due to sigmoid
            res = self.top_bottleneck(x)
            x = res["x"]
            z_top = x
            top_recon_loss = (torch.sum((x - x_preneck)**2, dim=1, keepdims=True)**.5).reshape((x_shape[0], 1, 1, 1))
            shortcut = x
            x = self.top_postneckFC(x)
            x = x.reshape(x_shape)
            x = torch.nn.LeakyReLU(negative_slope=0.01)(x)
            x = self.top_norm_interpost(x)
            x = self.top_postpostneckConv(x)
            x = x + self.top_post_shortcut_conv(shortcut.reshape((x.shape[0], -1, x.shape[2], x.shape[3])))

        firing_rates.append(res["expected_firing_rate"])
        real_firing_rates.append(res["real_firing_rate"])
        unnoised_z.append(res["unnoised_x"])
        z.append(z_top)

        # up paths

        x = self.forward_up_without_last(x, blocks, shortcut_multiplier = shortcut_multiplier)

        return x, firing_rates, real_firing_rates, unnoised_z, z, top_recon_loss

    def forward(self, x, shortcut_multiplier = 1.0):
        # average subtraction
        x = x - self.image_average_bias
        x = x / self.image_std_bias

        # main func
        x, firing_rates, real_firing_rates, unnoised_z, z, top_recon_loss = self.get_features(x, shortcut_multiplier)

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
            'log_var': self.last_logvar(x)/100.0,
            'firing_rate': firing_rates,
            'real_firing_rate': real_firing_rates,
            'unnoised_z': unnoised_z,
            'z': z,
            'top_recon_loss': top_recon_loss
        }

    def get_features(self, x, shortcut_multiplier = 1.0):
        return self.forward_without_last(x, shortcut_multiplier)


class CBAMCrossAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, reduction_ratio=16, norm_type="group", min_hidden=4):
        """
        F_g: gate channel num
        F_l: skip connection channel num
        F_int: internal channel num
        """
        super(CBAMCrossAttentionGate, self).__init__()
        
        # 1. ゲート信号(g)とスキップ特徴(x)を同じチャネル空間(F_int)に投影
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            make_norm2d(F_int, norm_type)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            make_norm2d(F_int, norm_type)
        )
        
        # 2. 融合した特徴に対するChannel Attention (CBAM形式)
        # v31: hidden width floored at min_hidden (was F_int // 16 = 0 at the 128^2 stage -> constant 0.5 weights)
        hidden = min(F_int, max(min_hidden, F_int // reduction_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(F_int, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, F_int, bias=False)
        )
        
        # 3. 融合した特徴に対するSpatial Attention (CBAM形式)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        # 最終的にスキップコネクション(F_l)のサイズに合わせる1x1畳み込みとシグモイド
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1) if norm_type == "group" else nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: [B, F_g, H, W] (Query)
        # x: [B, F_l, H, W] (Key/Value)
        
        # 互いの特徴を線形変換して足し合わせる (融合空間への投影)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_input = self.relu(g1 + x1) # [B, F_int, H, W]
        
        # --- Channel Attention ---
        # AvgPool と MaxPool を空間方向に適用
        avg_pool = torch.mean(psi_input, dim=[2, 3]) # [B, F_int]
        max_pool = torch.max(torch.max(psi_input, dim=2)[0], dim=2)[0] # [B, F_int]
        # MLPを通して足し合わせ、シグモイドでチャネル重みを計算
        channel_weight = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).unsqueeze(2).unsqueeze(3)
        psi_input = psi_input * channel_weight
        
        # --- Spatial Attention ---
        # チャネル方向に Avg と Max を計算して結合
        avg_out = torch.mean(psi_input, dim=1, keepdim=True) # [B, 1, H, W]
        max_out = torch.max(psi_input, dim=1, keepdim=True)[0] # [B, 1, H, W]
        spatial_input = torch.cat([avg_out, max_out], dim=1) # [B, 2, H, W]
        spatial_weight = torch.sigmoid(self.spatial_conv(spatial_input))
        psi_input = psi_input * spatial_weight
        
        # 最終的な注意度マップ（0~1）を計算
        attention_map = self.psi(psi_input) # [B, 1, H, W]
        
        # スキップコネクション特徴(x)にアテンションを適用
        return (x * attention_map, g * (1-attention_map))


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3, using_identity=False):
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

        self.in_size = in_size
        self.out_size = out_size

        self.using_identity = using_identity
        if using_identity and in_size != out_size:
            self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1)

    def forward(self, x):
        if self.using_identity:
            if self.in_size == self.out_size:
                out = x + self.block(x)
            else:
                out = self.shortcut(x) + self.block(x)
        else:
            out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group", islast=False, attention_gate=True, attention_selector=True, image_size=None, using_identity=False, norm_type="group"):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        elif up_mode == 'residual_autoencoder':
            self.up = ResidualUp(in_size, out_size)

        # the up path (out_size ch) is concatenated with the skip (out_size ch); in v31 in_size == 2*out_size always,
        # with a channel cap (v32 deep variants) in_size can equal out_size, so use 2*out_size explicitly
        self.conv_block = UNetConvBlock(2 * out_size, out_size, padding, norm=norm, using_identity=using_identity)

        if attention_gate:
            conv_in_size = out_size
            self.cbamcag = CBAMCrossAttentionGate(F_g=conv_in_size, F_l=conv_in_size, F_int=conv_in_size//2, norm_type=norm_type)
        else:
            self.cbamcag = None
            
        if attention_selector:
            self.attention_selector = attn_block(out_size, image_size ** 2)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        if bridge is None:
            bridge = torch.zeros_like(up)
        crop1 = self.center_crop(bridge, up.shape[2:])

        if self.cbamcag is not None:
            crop1, up = self.cbamcag(g=up, x=crop1)

        out = torch.cat([up, crop1], 1)
        
        out = self.conv_block(out)
        
        out = self.attention_selector(out)
        
        return out




if __name__ == '__main__':
    model = UNet()