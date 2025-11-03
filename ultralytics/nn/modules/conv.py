# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
'''
新加
'''
import torch.nn.functional as F

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    #新增ARD模块
    "A2DGLUConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
#=========================================== A2DGLUConv: Attention-based Residual Downsampling + CGLU ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

# 轻量通道注意：ECA（比 SE 更适合小模型）
class ECA(nn.Module):
    def __init__(self, c: int, k: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,1,C) -> 1D conv -> (B,C,1,1)
        y = self.pool(x).squeeze(-1).squeeze(-1)      # (B,C)
        y = self.conv(y.unsqueeze(1)).squeeze(1)      # (B,C)
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        return y

class A2DGLUConv(nn.Module):
    """
    A2DGLUConv (Aggressive small-object mAP50 booster)
    - 仅用于 stride=2 的下采样层：输出 (B, c2, H/2, W/2)
    - 设计：可学习轻量 Blur + 固定 Laplacian（双高频）；
           pixel_unshuffle 无损式下采；GLU × ECA × 边缘空间门；
           注回后做轻DWConv抑振铃；gamma 初始非零。
    """
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2,
                 gamma_init: float = 0.35, edge_gain: float = 1.5):
        super().__init__()
        assert s == 2, "A2DGLUConv 仅用于 stride=2 的下采样层"
        self.c1, self.c2 = c1, c2

        # —— Baseline 分支：保持原 YOLO 对齐/尺寸逻辑（稳定）
        # 依赖 Ultralytics 的 Conv(c1,c2,k=3,s=2) 已在该文件中定义
        self.base = Conv(c1, c2, k=k, s=2)

        # —— 轻量 Blur：DWConv，可学习（初始 3x3 binomial）
        self.blur = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        with torch.no_grad():
            k3 = torch.tensor([[1, 2, 1],
                               [2, 4, 2],
                               [1, 2, 1]], dtype=torch.float32) / 16.0
            self.blur.weight.copy_(k3.view(1, 1, 3, 3).repeat(c1, 1, 1, 1))
        self.blur.weight.requires_grad_(True)

        # —— 固定 Laplacian：强调边缘/角点
        self.lap = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        with torch.no_grad():
            k_lap = torch.tensor([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=torch.float32)
            self.lap.weight.copy_(k_lap.view(1, 1, 3, 3).repeat(c1, 1, 1, 1))
        self.lap.weight.requires_grad_(False)

        # —— 轻量空间注意（更偏向小目标/边缘）
        self.satt = nn.Sequential(
            nn.Conv2d(c1, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # —— 无损式下采样：pixel_unshuffle(2) 后 1x1 投影到 c2
        self.proj_unsh = nn.Conv2d(c1 * 4, c2, kernel_size=1, bias=False)

        # —— 边缘分量的 1x1 投影（作为边缘增强门的一部分）
        self.edge_proj = nn.Conv2d(c1, c2, kernel_size=1, bias=False)

        # —— GLU 门控
        self.glu_f = nn.Conv2d(c2, c2, 1, bias=True)
        self.glu_g = nn.Sequential(nn.Conv2d(c2, c2, 1, bias=True), nn.Sigmoid())

        # —— 轻量通道注意：ECA
        self.catt = ECA(c2)

        # —— 注回后轻滤波（抑制振铃/伪影）
        self.post_dw = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.BatchNorm2d(c2)
        )

        # —— 细节注回强度（初始非零：开局就参与）
        self.gamma = nn.Parameter(torch.ones(1, c2, 1, 1) * gamma_init)

        # —— 边缘增益（放 buffer，易于脚本中动态调）
        self.register_buffer("edge_gain", torch.tensor(float(edge_gain), dtype=torch.float32))

    @staticmethod
    def _even_crop(x: torch.Tensor) -> torch.Tensor:
        """保证 H、W 为偶数，避免 pixel_unshuffle 奇数边报错。"""
        _, _, h, w = x.shape
        h2 = h - (h % 2)
        w2 = w - (w % 2)
        if h2 != h or w2 != w:
            x = x[..., :h2, :w2]
        return x

    @torch.no_grad()
    def set_warmup(self, progress: float,
                   gamma_range=(0.10, 0.35), edge_range=(0.8, 1.8)):
        """
        可选：温启动（0~1）在训练循环中调用：
            m.set_warmup(min(epoch/T, 1.0))
        """
        g0, g1 = gamma_range
        e0, e1 = edge_range
        p = float(max(0.0, min(1.0, progress)))
        self.gamma.data[:] = g0 + (g1 - g0) * p
        self.edge_gain[:] = torch.tensor(e0 + (e1 - e0) * p, dtype=self.edge_gain.dtype, device=self.edge_gain.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # —— Baseline 下采样（稳定主干）
        y_base = self.base(x)  # (B,c2,H/2,W/2)

        # —— 高频分量：弱抗混叠 + Laplacian
        blur = self.blur(x)
        hf   = torch.tanh(x - blur)   # 抗混叠后的高频
        lap  = torch.tanh(self.lap(x))  # 强边缘

        # —— 空间注意
        a = self.satt(x)                         # (B,1,H,W)
        keep = (hf + lap) * a                    # 叠加强化

        # —— 无损式下采 + 投影
        keep = self._even_crop(keep)
        p_un = F.pixel_unshuffle(keep, 2)        # (B, c1*4, H/2, W/2)
        p    = self.proj_unsh(p_un)              # (B, c2,   H/2, W/2)

        # —— GLU
        ph = self.glu_f(p) * self.glu_g(p)       # (B,c2,H/2,W/2)

        # —— 边缘增强门（spatial × channel）
        s_down = F.avg_pool2d(a, 2, 2)           # (B,1,H/2,W/2)
        e      = self.edge_proj(F.avg_pool2d(lap, 2, 2))  # (B,c2,H/2,W/2)
        boost  = 1.0 + self.edge_gain * torch.tanh(e) * s_down
        boost  = torch.clamp(boost, 0.5, 2.5)    # 防极端放大

        # —— 通道门（ECA）
        c_gate = self.catt(ph)                   # (B,c2,1,1)

        # —— 细节注回 + 轻滤波
        detail = self.post_dw(ph * boost * c_gate)
        out = y_base + self.gamma * detail
        return out

#=======================================================================================================================
                                                    #RGCU
#=======================================================================================================================
# ====== ultralytics/nn/modules/conv.py 里替换/新增这一段 ======
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从 util_converse 导入上采样核心
from .util_converse import ConverseUp2d

class ECA(nn.Module):
    """轻量通道注意力（不降维），更稳定、收敛快"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) / gamma) + b))
        k = k if k % 2 else k + 1
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k // 2), bias=False)

    def forward(self, x):  # x:(B,C,1,1)
        y = self.avg(x)                      # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)    # (B,1,C)
        y = self.conv(y)
        y = y.transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
        return torch.sigmoid(y)


class RGCU(nn.Module):
    """
    RGCU v2（稳定增强版）: Risk-Gated Converse Upsample (×2, 频域反卷积)
      - Path1: 最近邻 + 1×1（保守、稳）
      - Path2: ConverseUp2d（精细，上采样在 FP32 计算，数值更稳）
      - 门控: S_up（空间） + G_c（通道, 基于 U1+U2 的融合统计）+ R（一致性, 归一化后计算）
             M = sigmoid((a·S_up + b·G_c + c·R) / τ)
    """
    def __init__(self, c1: int, c2: int, scale: int = 2):
        super().__init__()
        assert scale == 2, "当前实现默认 ×2"
        self.c1, self.c2, self.scale = c1, c2, scale

        # 空间风险
        self.s_dw = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.s_pw = nn.Conv2d(c1, 1, 1, 1, 0, bias=True)

        # Path1：最近邻 + 1×1
        self.near_proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)

        # Path2：通道对齐 + 频域上采样 + 回对齐
        mid = max(c1, c2)
        self.pre_align  = nn.Conv2d(c1, mid, 1, bias=False) if c1 != mid else nn.Identity()
        self.learn_up   = ConverseUp2d(mid, mid, scale=scale, kernel_size=3, padding=2)
        self.post_align = nn.Conv2d(mid, c2, 1, bias=False) if c2 != mid else nn.Identity()

        # 通道注意力：ECA（更稳的统计：用 U1+U2 融合）
        self.eca = ECA(c2)

        # 一致性门：先对齐统计（防幅值误伤）
        self.r_norm = nn.GroupNorm(num_groups=min(32, c2), num_channels=c2, eps=1e-5, affine=False)

        # 联合门控参数（更果断的初值 + 可暖启的温度）
        self.a_raw   = nn.Parameter(torch.tensor(0.6))  # >=0 via softplus
        self.b_raw   = nn.Parameter(torch.tensor(0.6))
        self.c_raw   = nn.Parameter(torch.tensor(0.3))
        self.tau_raw = nn.Parameter(torch.tensor(1.0))  # 训练中可降到 ~0.5

        # 输出平滑：DW3×3 + PW1×1（抑振铃/伪影）
        self.smooth_dw = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.smooth_pw = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)

    @torch.no_grad()
    def set_tau_warmup(self, progress: float, start=1.0, end=0.5):
        """
        可选：在训练循环里调用，progress ∈ [0,1]，把 τ 从 start 线性降到 end
        """
        p = float(max(0.0, min(1.0, progress)))
        self.tau_raw.data = torch.tensor(start + (end - start) * p, device=self.tau_raw.device)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # 空间风险
        S = torch.sigmoid(self.s_pw(self.s_dw(y)))                        # (B,1,H,W)
        S_up = F.interpolate(S, scale_factor=self.scale, mode="nearest")  # (B,1,2H,2W)

        # Path1（保守）
        U1 = F.interpolate(y, scale_factor=self.scale, mode="nearest")    # (B,C1,2H,2W)
        U1 = self.near_proj(U1)                                           # (B,C2,2H,2W)

        # Path2（精细，上采样在 FP32 计算更稳）
        z = self.pre_align(y)
        if torch.is_autocast_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                U2 = self.learn_up(z.float())
        else:
            U2 = self.learn_up(z)
        U2 = self.post_align(U2).to(z.dtype)                              # (B,C2,2H,2W)

        # 通道门：用 U1+U2 的融合统计，降低对 U2 抖动的耦合
        G_c = self.eca(0.5 * (U1 + U2))                                   # (B,C2,1,1)

        # 一致性门：归一化后计算，且 detach U1 防止互相追逐
        U1_n = self.r_norm(U1)
        U2_n = self.r_norm(U2)
        R = torch.mean(torch.abs(U2_n - U1_n.detach()), dim=1, keepdim=True)  # (B,1,2H,2W)
        R = torch.sigmoid(R)

        # 联合门控 + 温度
        a   = F.softplus(self.a_raw)
        b   = F.softplus(self.b_raw)
        c   = F.softplus(self.c_raw)
        tau = F.softplus(self.tau_raw) + 1e-6

        M = torch.sigmoid((a * S_up + b * G_c + c * R) / tau)             # (B,C2,2H,2W)

        out = (1.0 - M) * U1 + M * U2
        out = self.smooth_pw(self.smooth_dw(out))
        return out
