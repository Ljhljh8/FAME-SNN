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

        x = self.mlp1_lif(x, alpha)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x)
        x = self.drop1(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x, alpha)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x)
        x = self.drop2(x).reshape(T, B, C, H, W)
        return x

def hartley2(x, operation):
    """Orthonormal Hartley components/inverse or modular frequency reflection."""
    if operation == 'reflect':
        return torch.roll(torch.flip(x, dims=(-2, -1)), shifts=(1, 1), dims=(-2, -1))
    if operation not in ('forward', 'inverse'):
        raise ValueError('unknown Hartley operation: %s' % operation)
    spectrum = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    if operation == 'forward':
        return spectrum.real, -spectrum.imag
    return spectrum.real - spectrum.imag


class ASFF(nn.Module):
    """Revised Hartley GIF, with the dataset's original hidden capacity."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, hidden_size_factor=1):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.hidden_size = dim
        self.num_blocks = num_heads
        self.block_size = dim // num_heads
        self.hidden_size_factor = 4
        self.scale = 0.002
        self.softshrink = 0.06
        self.w1 = nn.Parameter(self.scale * torch.randn(
            1, num_heads, self.block_size, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(
            1, num_heads, self.block_size * self.hidden_size_factor, self.block_size))
        self.bn1 = nn.BatchNorm2d(dim * self.hidden_size_factor)
        self.bn2 = nn.BatchNorm2d(dim)
        # BNinitial / neutral-init, applied after the final kernel BN is built.
        nn.init.constant_(self.bn2.weight, 0.01)
        nn.init.constant_(self.bn2.bias, 1.0 + self.softshrink)
        self.input_lif = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.kernel_input_lif = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.feature_even_lif = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.feature_odd_lif = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.hidden_activation = neuron.STHLIFNode(step_mode='m', tau=tau_thr, v_threshold=0.5, detach_reset=True, backend='cupy')

    def forward(self, x, alpha):
        T, B, C, H, W = x.shape
        dtype = x.dtype
        # Hartley operations and the learned frequency response use FP32 under AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            source = self.input_lif(x.float(), alpha)
            even, odd = hartley2(source, 'forward')
            kernel_source = self.kernel_input_lif(even + odd, alpha)
            feature_even = self.feature_even_lif(even, alpha)
            feature_odd = self.feature_odd_lif(odd, alpha)
            blocked = kernel_source.reshape(T * B, self.num_blocks, self.block_size, H, W)
            stage1 = torch.einsum('bkihw,kio->bkohw', blocked, self.w1[0])
            stage1 = self.bn1(stage1.flatten(1, 2)).reshape(T, B, C * self.hidden_size_factor, H, W)
            hidden = self.hidden_activation(stage1, alpha).reshape(
                T * B, self.num_blocks, self.block_size * self.hidden_size_factor, H, W)
            stage2 = torch.einsum('bkihw,kio->bkohw', hidden, self.w2[0])
            kernel = F.softshrink(self.bn2(stage2.flatten(1, 2)), self.softshrink).reshape(T, B, C, H, W)
            # H(-K)[u,v] = H(K)[(-u) mod H, (-v) mod W].
            reflected = hartley2(kernel, 'reflect')
            mixed = kernel * feature_even + reflected * feature_odd
            return hartley2(mixed, 'inverse').to(dtype)
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
def split_contiguous_windows(x, num_windows):
    """[T,B,C,H,W] -> [T/num_windows,num_windows*B,C,H,W]."""
    T, B, C, H, W = x.shape
    if num_windows < 1 or T % num_windows:
        raise ValueError("T must be divisible by the window count")
    window_length = T // num_windows
    return (x.reshape(num_windows, window_length, B, C, H, W)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(window_length, num_windows * B, C, H, W))


def restore_temporal_order(x, num_windows):
    """[window_length,num_windows*B,C,H,W] -> [T,B,C,H,W]."""
    window_length, window_batch, C, H, W = x.shape
    if num_windows < 1 or window_batch % num_windows:
        raise ValueError("window batch must be divisible by the window count")
    B = window_batch // num_windows
    return (x.reshape(window_length, num_windows, B, C, H, W)
            .permute(1, 0, 2, 3, 4, 5)
            .reshape(window_length * num_windows, B, C, H, W))


class MS_STF(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super().__init__()
        mlp_ratio = 2
        self.embeddim = out_channel
        
        self.multi_T = [8, 4, 2 ,1]
        basethreshold = 0.5
        basetau = 1.75
        beta_tau = 1.0
        beta_th = 0.5
        self.thresholds = [basethreshold + beta_th * math.log(idx + 1) for idx, t in enumerate(self.multi_T)]
        self.tau_thr = [basetau + beta_tau * math.log(idx + 1) for idx, t in enumerate(self.multi_T)]

        # self.thresholds.reverse()
        # self.MSSF = nn.ModuleList([nn.Sequential(nn.Conv2d(in_ch, out_channel // mlp_ratio, kernel_size=1, bias=False),
        #                                          nn.BatchNorm2d(out_channel // mlp_ratio)) for in_ch in in_channel_list])
        self.MSSF = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=1, padding=2, groups=in_ch, bias=False),
                           nn.Conv2d(in_ch, out_channel * mlp_ratio, kernel_size=1, bias=False),
                           nn.BatchNorm2d(out_channel * mlp_ratio)) for in_ch in in_channel_list])
        # self.mssf_bn = nn.ModuleList([nn.BatchNorm2d(out_channel//mlp_ratio) for in_ch in range(len(in_channel_list)-1)])

        self.MSTF_lif = nn.ModuleList(
            [neuron.STHLIFNode(step_mode='m', tau=self.tau_thr[i], v_threshold=self.thresholds[i], detach_reset=True, backend='cupy')
             for i in range(len(in_channel_list))])
        # self.MSTF = nn.ModuleList([nn.Sequential(nn.Linear(out_channel // mlp_ratio, out_channel),
        #                                          nn.LayerNorm(out_channel)) for _ in in_channel_list])
        self.MSTF = nn.ModuleList([nn.Sequential(
            # Dcls2d(out_channel, out_channel, kernel_count=25, dilated_kernel_size=5,padding=2,stride=1,
            #        groups=out_channel, bias=False),
            nn.Conv1d(out_channel * mlp_ratio, out_channel * mlp_ratio, kernel_size=3, padding=1, stride=1, groups=out_channel, bias=False),
            nn.Conv1d(out_channel * mlp_ratio, self.embeddim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(self.embeddim)) for _ in in_channel_list])

        self.mp1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.mp2 = torch.nn.MaxPool2d(kernel_size=6, stride=4, padding=1, dilation=1)
        self.MSSTF_bn = nn.BatchNorm2d(out_channel)

    def forward(self, x, alpha):
        head_output = []

        T, B, _, H, W = x[-1].shape
        if any(n < 1 or T % n for n in self.multi_T):
            raise ValueError("T must be divisible by every MS-STF window count")
        add_pre2corent = self.MSSF[-1](x[-1].flatten(0, 1))
        tempro_fusion = split_contiguous_windows(
            add_pre2corent.reshape(T, B, -1, H, W), self.multi_T[-1]
        )
        tempro_fusion = self.MSTF_lif[-1](tempro_fusion, alpha)

        tempro_fusion = tempro_fusion.permute(3, 4, 1, 2, 0).contiguous()  # H*W*B, C, T
        tempro_fusion = self.MSTF[-1](tempro_fusion.flatten(0, 2))  # H*W*B, C, T
        tempro_fusion = tempro_fusion.view(H, W, B * self.multi_T[-1], -1, T // self.multi_T[-1]).permute(4, 2, 3, 0,
                                                                                                          1).contiguous()
        tempro_fusion = restore_temporal_order(tempro_fusion, self.multi_T[-1])
        head_output.append((tempro_fusion))
        for i in range(len(x) - 2, -1, -1):
            corent_inner = (self.MSSF[i](x[i].flatten(0, 1)))
            if corent_inner.shape[2:] != add_pre2corent.shape[2:]:
                add_pre2corent = F.interpolate(add_pre2corent, size=corent_inner.shape[2:])
            add_pre2corent = (add_pre2corent + corent_inner)
            tempro_fusion = add_pre2corent
            if i == 1:
                tempro_fusion = self.mp1(tempro_fusion)
            elif i == 0:
                tempro_fusion = self.mp2(tempro_fusion)
            tempro_fusion = split_contiguous_windows(
                tempro_fusion.reshape(T, B, -1, H, W), self.multi_T[i]
            )
            tempro_fusion = self.MSTF_lif[i](tempro_fusion, alpha)

            tempro_fusion = tempro_fusion.permute(3, 4, 1, 2, 0).contiguous()  # H*W*B, C, T
            tempro_fusion = self.MSTF[i](tempro_fusion.flatten(0, 2))  # H*W*B, C, T
            tempro_fusion = tempro_fusion.view(H, W, B * self.multi_T[i], -1, T // self.multi_T[i]).permute(4, 2, 3, 0,
                                                                                                            1).contiguous()
            tempro_fusion = restore_temporal_order(tempro_fusion, self.multi_T[i])
            head_output.append(tempro_fusion)

        x = torch.stack((list((head_output))), dim=0).reshape(-1, T, B, self.embeddim, H, W).contiguous()

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

    def forward(self, x, train_lenloader=0, epoch=0, batch_idx=0):
        # Twice-per-epoch quadratic STHOS, determined by epoch/batch (resume safe).
        if not self.training:
            self.alpha = 0.0
        else:
            endpoint = self.STH_epoch
            if endpoint is None or endpoint == 0 or epoch >= endpoint:
                self.alpha = 0.0
            else:
                if endpoint < 0 or epoch < 0 or train_lenloader < 1 or not 0 <= batch_idx < train_lenloader:
                    raise ValueError("invalid STHOS epoch, endpoint, or batch position")
                half = int(batch_idx >= (train_lenloader + 1) // 2)
                initial = getattr(self, 'alpha_init', 1.0)
                initial = 1.0 if initial is None else float(initial)
                if not math.isfinite(initial) or not 0.0 <= initial <= 1.0:
                    raise ValueError("alpha_init must be finite and within [0, 1]")
                self.alpha = initial * (1.0 - (2 * epoch + half) / (2.0 * endpoint)) ** 2
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