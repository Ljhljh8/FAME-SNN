import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode as MultiStepLIFNode
from STHOS import neuron
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from timm.models import create_model
import math
__all__ = ['MS_FAST']
tau_thr = 2.0
import torch
import torch.fft


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp_bn = nn.BatchNorm2d(in_features)
        self.mlp1_lif = neuron.STHLIFNode(step_mode='m',tau=tau_thr, v_threshold=1.0, detach_reset=True, backend='cupy')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)
        self.drop1 = nn.Dropout(drop)

        self.mlp2_lif = neuron.STHLIFNode(step_mode='m',tau=tau_thr, v_threshold=1.0, detach_reset=True, backend='cupy')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)
        self.drop2 = nn.Dropout(drop)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x, alpha):
        T, B, C, H, W = x.shape

        x = self.mlp_bn(x.flatten(0,1)).reshape(T, B, C, H, W)
        x = self.mlp1_lif(x, alpha)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x)
        x = self.drop1(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x, alpha)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x)
        x = self.drop2(x).reshape(T, B, C, H, W)
        return x

class ASFF(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.hidden_size = dim
        self.num_blocks = num_heads
        self.block_size = self.hidden_size // self.num_blocks

        self.scale = 0.002
        self.th = 0.5

        self.w1 = nn.Parameter(
            self.scale * torch.randn(1, self.num_blocks, self.block_size, self.block_size))

        self.ASSF_bn = nn.BatchNorm2d(dim)
        self.act = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=self.th, detach_reset=True, backend='cupy')
        self.act11 = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=self.th, detach_reset=True, backend='cupy')
        self.bn1_1 = nn.BatchNorm2d(dim )
        self.act12 = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=self.th, detach_reset=True, backend='cupy')
        self.bn1_2 = nn.BatchNorm2d(dim )
        self.act21 = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=self.th, detach_reset=True, backend='cupy')
        self.bn2_1 = nn.BatchNorm2d(dim)
        self.act22 = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=self.th, detach_reset=True, backend='cupy')
        self.bn2_2 = nn.BatchNorm2d(dim)
        self.drop_atten = nn.Dropout(attn_drop)


        self.ASSF1_bn = nn.BatchNorm2d(dim)


    def hartley_transform(self, x):
        # Utilize the relationship between Fourier Transform and Hartley Transform for the fast Hartley Transform.
        return torch.fft.fft2((x), dim = (3, 4), norm="ortho")

    def inverse_hartley_transform(self, x):
        # Since the Hartley Transform is self-inverse, we can directly apply the Hartley Transform again to obtain the inverse transform.
        x = self.hartley_transform(x)
        return (x.real - x.imag)

    def forward(self, x, alpha):
        T, B, C, H, W = x.shape
        dtype = x.dtype
        x = x.float()
        x = self.ASSF_bn(x.flatten(0, 1)).reshape(T, B, C, H, W)
        x = self.act(x, alpha)
        x = self.hartley_transform(x)

        origin_ffted_so = self.act21(x.real, alpha)
        origin_ffted_se = self.act22(x.imag, alpha)

        x.real = origin_ffted_so
        x.imag = origin_ffted_se
        x = x.flatten(0, 1).reshape(T*B, self.num_blocks, self.block_size, x.shape[3], x.shape[4])
        o1_real = self.act11(
            self.bn1_1((torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
                        torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0])).flatten(1, 2)).reshape(T, B,self.num_blocks,self.block_size,x.shape[3],x.shape[4])
        , alpha).flatten(0, 1)
        o1_imag = self.act12(
            self.bn1_2((torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
                        torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0])).flatten(1, 2)).reshape(T, B,self.num_blocks,self.block_size,
                                                                                                                      x.shape[3],x.shape[4]), alpha).flatten(0, 1)



        o2_real = (self.bn2_1((torch.einsum('bkihw,kio->bkohw', o1_real, self.w1[0]) - \
                               torch.einsum('bkihw,kio->bkohw', o1_imag, self.w1[0])).flatten(1, 2))
                             .reshape(T, B,self.num_blocks,self.block_size,x.shape[3],x.shape[4]))
        o2_imag = (self.bn2_2((torch.einsum('bkihw,kio->bkohw', o1_imag, self.w1[0]) + \
                               torch.einsum('bkihw,kio->bkohw', o1_real, self.w1[0])).flatten(1, 2))
                             .reshape(T, B,self.num_blocks,self.block_size,x.shape[3],x.shape[4]))

        o2_real = F.softshrink(o2_real.reshape(T, B, C, o2_real.shape[4], o2_real.shape[5]), 0.06)
        o2_imag = F.softshrink(o2_imag.reshape(T, B, C, o2_imag.shape[4], o2_imag.shape[5]), 0.06)

        o2_fp = o2_real - o2_imag
        o2_fn = o2_real + o2_imag
        x = ((origin_ffted_so * o2_fp) - (origin_ffted_se * o2_fn)).float()

        x = self.inverse_hartley_transform(x)

        x = x.type(dtype)
        x = self.ASSF1_bn(x.flatten(0, 1)).reshape(T, B, C, H, W)
        x = self.drop_atten(x)
        return x
# from spikingjelly.visualizing import plot_2d_heatmap
class GIF(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop_mlp=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.attn = ASFF(dim, num_heads=num_heads,
                                           attn_drop=attn_drop, proj_drop=drop_mlp, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop_mlp)
    def forward(self, x, alpha):

        x = x + self.drop_path(self.attn(x, alpha))
        x = x + self.drop_path(self.mlp(x, alpha))
        return x
class MS_STF(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super().__init__()
        mlp_ratio = 1
        self.embeddim = out_channel

        self.multi_T = [8, 4, 2 ,1]
        basethreshold = 0.5
        basetau = 1.75
        beta_tau = 1.0
        beta_th = 0.5
        self.thresholds = [basethreshold + beta_th * math.log(idx + 1) for idx, t in enumerate(self.multi_T)]
        self.tau_thr = [basetau + beta_tau * math.log(idx + 1) for idx, t in enumerate(self.multi_T)]

        # self.thresholds.reverse()
        self.MSSF = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, out_channel // mlp_ratio, kernel_size=1, bias=False),
                                                 nn.BatchNorm2d(out_channel // mlp_ratio)) for in_ch in in_channel_list])
        self.mssf_bn = nn.ModuleList([nn.BatchNorm2d(out_channel//mlp_ratio) for in_ch in range(len(in_channel_list)-1)])

        self.MSTF_lif = nn.ModuleList(
            [neuron.STHLIFNode(step_mode='m', tau=self.tau_thr[i], v_threshold=self.thresholds[i], detach_reset=True, backend='cupy')
             for i in range(len(in_channel_list))])
        self.MSTF = nn.ModuleList([nn.Sequential(nn.Linear(out_channel // mlp_ratio, out_channel),
                                                 nn.LayerNorm(out_channel)) for _ in in_channel_list])

        self.mp1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=6, stride=4, padding=1, dilation=1)
        self.MSSTF_bn = nn.BatchNorm2d(out_channel)

    def forward(self, x, alpha):
        head_output = []

        T, B, _, H, W = x[-1].shape
        add_pre2corent = self.MSSF[-1](x[-1].flatten(0, 1))
        tempro_fusion = self.MSTF_lif[-1](add_pre2corent.reshape(T//self.multi_T[-1], self.multi_T[-1]*B, -1, H, W), alpha)
        tempro_fusion = tempro_fusion.reshape(T, B, -1, H, W)

        tempro_fusion = tempro_fusion.permute(3, 4, 1, 0, 2).contiguous()    # H, W, B, T, C, or  # H, W, B*i, T/i, C,
        tempro_fusion = self.MSTF[-1](tempro_fusion.flatten(0, 2))           # H, W, B, T, C, or  # H, W, B*i, T/i, C,
        head_output.append((tempro_fusion))
        for i in range(len(x) - 2, -1, -1):
            corent_inner = (self.MSSF[i](x[i].flatten(0, 1)))
            if corent_inner.shape[2:] != add_pre2corent.shape[2:]:
                add_pre2corent = F.interpolate(add_pre2corent, size=corent_inner.shape[2:])
            add_pre2corent = self.mssf_bn[i](add_pre2corent + corent_inner)
            tempro_fusion = add_pre2corent
            if i == 1:
                tempro_fusion = self.mp1(tempro_fusion)
            elif i == 0:
                tempro_fusion = self.mp2(tempro_fusion)
            tempro_fusion = self.MSTF_lif[i](tempro_fusion.reshape(T//self.multi_T[i], self.multi_T[i]*B, -1, H, W), alpha)
            tempro_fusion = tempro_fusion.reshape(T, B, -1, H, W)

            tempro_fusion = tempro_fusion.permute(3, 4, 1, 0, 2).contiguous()   # H, W, B, T, C
            tempro_fusion = self.MSTF[i](tempro_fusion.flatten(0, 2))
            head_output.append(tempro_fusion)
        x = torch.stack((list((head_output))), dim=0).reshape(-1, H, W, B, T, self.embeddim)
        x = x.permute(0, 4, 3, 5, 1, 2).contiguous()
        x = torch.mean(x, dim=0)
        return x

class SpikingTokenizer(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256, T=16):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.T = T

        self.multi_T = [1, 1, 1, 1, 1]
        self.thresholds = [1.0 for t in self.multi_T]
        self.tau_thr = [2.0 for t in self.multi_T]

        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj1_lif = neuron.STHLIFNode(step_mode='m',tau=self.tau_thr[0], v_threshold=self.thresholds[0], detach_reset=True, backend='cupy')
        self.proj1_conv = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims//4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj2_lif = neuron.STHLIFNode(step_mode='m',tau=self.tau_thr[1], v_threshold=self.thresholds[1], detach_reset=True, backend='cupy')
        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims//2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj3_lif = neuron.STHLIFNode(step_mode='m',tau=self.tau_thr[2], v_threshold=self.thresholds[2], detach_reset=True, backend='cupy')
        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj4_lif = neuron.STHLIFNode(step_mode='m',tau=self.tau_thr[3], v_threshold=self.thresholds[3], detach_reset=True, backend='cupy')
        self.proj4_conv = nn.Conv2d(embed_dims, embed_dims*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dims*2)
        self.proj5_lif = neuron.STHLIFNode(step_mode='m', tau=self.tau_thr[4], v_threshold=self.thresholds[4], detach_reset=True, backend='cupy')

        self.MS_STF = MS_STF([embed_dims//4, embed_dims//2, embed_dims, embed_dims*2], embed_dims,
                             )



    def forward(self, x, alpha):
        T, B, C, H, W = x.shape
        FPN_input = []

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x)
        x = (self.maxpool(x)).reshape(T//self.multi_T[0], self.multi_T[0]*B, -1, 64, 64)

        x = self.proj1_lif(x, alpha)
        x = self.proj1_conv(x.flatten(0,1))
        x = self.proj1_bn(x)
        x = (self.maxpool1(x)).reshape(T//self.multi_T[1], self.multi_T[1]*B, -1, 32, 32)

        x = self.proj2_lif(x,alpha)
        FPN_input.append(x)
        x = self.proj2_conv(x.flatten(0, 1))
        x = self.proj2_bn(x)
        x = (self.maxpool2(x)).reshape(T//self.multi_T[2], self.multi_T[2]*B, -1, 16, 16)

        x = self.proj3_lif(x, alpha)
        x = x.reshape(T, B, -1, 16, 16)
        FPN_input.append(x)
        x = self.proj3_conv(x.flatten(0, 1))
        x = self.proj3_bn(x)
        x = (self.maxpool3(x)).reshape(T//self.multi_T[3], self.multi_T[3]*B, -1, H//16, W//16)

        x = self.proj4_lif(x, alpha)
        x = x.reshape(T, B, -1, H // 16, W // 16)
        FPN_input.append(x)

        x = self.proj4_conv(x.flatten(0, 1))
        x = self.proj4_bn(x).reshape(T//self.multi_T[4], self.multi_T[4]*B, -1, H//16, W//16)
        x = self.proj5_lif(x, alpha)
        FPN_input.append(x)

        x = self.MS_STF(FPN_input, alpha)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class vit_snn(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=96, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4],
                 drop_mlp=0., attn_drop_rate=0., drop_path_rate=0., drop_out_sps=0.,

                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=16, STHOS=None, alpha_init=None,
                 pretrained_cfg=None, pretrained_cfg_overlay=None, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.iter_num = 0


        self.alpha = 0.
        self.STH_epoch = STHOS

        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        self.drop_out_sps = DropPath(drop_out_sps)
        patch_embed = SpikingTokenizer(img_size_h=img_size_h,
                          img_size_w=img_size_w,
                          patch_size=patch_size,
                          in_channels=in_channels,
                          embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        block = nn.ModuleList([GIF(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, drop_mlp=drop_mlp, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])
        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, ):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x, (H, W) = patch_embed(x, self.alpha)
        # x = self.drop_out_sps(x)
        for blk in block:
            x = blk(x, self.alpha)
        return x.flatten(3).mean(3)

    def forward(self, x, train_lenloader=0, epoch=0):
        train_lenloader = train_lenloader
        step_size = (train_lenloader // 2)
        if self.training and epoch < self.STH_epoch:
            decay_factor = 1.0 - self.iter_num / (self.STH_epoch * train_lenloader)
            if self.iter_num % step_size == 0:
                if decay_factor > 0.0:
                    self.alpha = math.pow(decay_factor, 2.0)
                    print(self.iter_num)
                    print(self.alpha)
                else:
                    self.alpha = 0.0
            self.iter_num += 1
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head((x.mean(0)))
        return x


@register_model
def MS_FAST(pretrained=False, **kwargs):
    model = vit_snn(
        # patch_size=16, embed_dims=192, num_heads=6, mlp_ratios=4,
        # in_channels=2, num_classes=11,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=2, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


from timm.models import create_model

if __name__ == '__main__':
    x = torch.randn(1, 1, 2, 128, 128).cuda()
    model = create_model(
        'MS_FAST',
        pretrained=False,
        drop_rate=0,
        drop_path_rate=0.3,
        drop_block_rate=None,
    ).cuda()
    model.eval()
    y = model(x)
    print(y.shape)
    print('Test Good!')