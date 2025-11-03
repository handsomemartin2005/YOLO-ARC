# ultralytics/nn/modules/util_converse.py
# Converse2D_Wiener: 频域反卷积上采样（Wiener 正则 + 径向低通）
# ConverseUp2d: 与工程内原接口保持一致的包装（类名大小写很重要：U 要大写）

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "Converse2D_Wiener",
    "ConverseUp2d",
]


class Converse2D_Wiener(nn.Module):
    """
    频域反卷积上采样（×scale），每通道独立核（depthwise-like）
    - 内部统一 FP32（AMP 安全，避免 ComplexHalf 问题）
    - replicate 边界，避免 circular 伪影污染边缘小目标
    - Wiener 正则 + 学习型径向低通（抑振铃，更适合小目标边缘）
    约束：当前实现要求 in_channels == out_channels（外侧用 1×1 对齐即可）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        scale: int = 2,
        padding: int = 2,
        padding_mode: str = "replicate",
        eps: float = 1e-5,
        init_cutoff: float = 0.35,
        init_order: float = 2.0,
        init_lambda: float = 1e-3,
    ) -> None:
        super().__init__()
        assert in_channels == out_channels, "Converse2D_Wiener 目前约束 Cin==Cout，请在外侧用1×1处理通道对齐。"
        self.c = in_channels
        self.k = kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        # 每通道独立核（初始化 softmax 归一，避免能量漂移）
        self.weight = nn.Parameter(torch.randn(1, self.c, self.k, self.k))
        with torch.no_grad():
            w = F.softmax(self.weight.view(1, self.c, -1), dim=-1).view_as(self.weight)
            self.weight.copy_(w)

        # Wiener 正则强度 λ（训练中自适应）
        self.lam_raw = nn.Parameter(torch.tensor(float(init_lambda)))
        # 学习型径向低通参数：cutoff ∈ (0.05, 0.5)，order ≥ 1
        self.cutoff = nn.Parameter(torch.tensor(float(init_cutoff)))
        self.order = nn.Parameter(torch.tensor(float(init_order)))

    @staticmethod
    def _p2o(psf: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        """
        PSF -> OTF（中心对齐）
        返回复数频谱（complex64/complex32，依据输入 dtype 决定）
        """
        otf = torch.zeros(psf.shape[:-2] + shape, dtype=psf.dtype, device=psf.device)
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        # 半尺寸 roll，保证核中心与频域原点对齐
        otf = torch.roll(otf, shifts=(-psf.shape[-2] // 2, -psf.shape[-1] // 2), dims=(-2, -1))
        return torch.fft.fftn(otf, dim=(-2, -1))

    @staticmethod
    def _radial_mask(
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
        cutoff: torch.Tensor,
        order: torch.Tensor,
    ) -> torch.Tensor:
        """学习型径向 Butterworth 低通掩码（抑制超高频振铃）"""
        fy = torch.fft.fftfreq(h, d=1.0, device=device)
        fx = torch.fft.fftfreq(w, d=1.0, device=device)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        r = torch.sqrt(xx * xx + yy * yy).to(dtype)
        c = torch.clamp(cutoff, 0.05, 0.5)
        n = torch.clamp(order, 1.0, 8.0)
        mask = 1.0 / (1.0 + (r / (c + 1e-8)).pow(2.0 * n))
        return mask  # (H, W), real

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 频域计算统一用 FP32，避免 ComplexHalf 未实现的问题
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        # 边界复制 padding（更适合检测）
        if self.padding > 0:
            x = F.pad(
                x,
                [self.padding, self.padding, self.padding, self.padding],
                mode=self.padding_mode,
            )

        B, C, H, W = x.shape
        s = self.scale

        # s-fold 插零上采样（零保持对齐，避免半像素偏移）
        z = torch.zeros((B, C, H * s, W * s), dtype=x.dtype, device=x.device)
        z[..., 0::s, 0::s] = x

        # 频域核 & 输入频谱
        FB = self._p2o(self.weight.to(x.dtype), (H * s, W * s))  # (1,C,Hs,Ws), complex
        FBC = torch.conj(FB)
        F2B = torch.abs(FB) ** 2
        Y = torch.fft.fftn(z, dim=(-2, -1))  # (B,C,Hs,Ws), complex

        # Wiener 反卷积：H* Y / (|H|^2 + λ)
        lam = F.softplus(self.lam_raw) + self.eps
        FX = FBC * Y / (F2B + lam)

        # 径向低通抑振铃
        LP = self._radial_mask(H * s, W * s, x.device, x.dtype, self.cutoff, self.order)  # (Hs,Ws)
        FX = FX * LP  # broadcast

        # 反变换 & 去 padding
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))
        if self.padding > 0:
            t = self.padding * s
            out = out[..., t:-t, t:-t]

        return out.to(orig_dtype)


class ConverseUp2d(nn.Module):
    """
    与工程原有接口保持一致的包装：
      ConverseUp2d(c1, c2, scale=2, kernel_size=3, padding=2, padding_mode='replicate')
    说明：当前实现为稳定性约束 c1==c2；若不等，请在外侧用 1×1 卷积处理通道对齐。
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        scale: int = 2,
        kernel_size: int = 3,
        padding: int = 2,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()
        assert c1 == c2, "ConverseUp2d 目前约束 c1==c2（可在外侧用1×1先对齐通道）"
        self.up = Converse2D_Wiener(
            c1,
            c2,
            kernel_size=kernel_size,
            scale=scale,
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)
