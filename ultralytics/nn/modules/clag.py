# ultralytics/nn/modules/clag.py
# Cross-Layer Attention Guidance (CLAG) for YOLOv8 neck fusion
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TapAttention", "CLAG"]

class TapAttention(nn.Module):
    """
    从浅层特征抽取 1 通道注意力图 M \in [0,1]（Tap）。
    设计：1x1 降/对齐 -> BN+SiLU -> 1x1 -> Sigmoid
    输入:  x_shallow: (B, C_s, Hs, Ws)
    输出:  M: (B, 1, Hs, Ws)
    """
    def __init__(self, c_in, mid_ratio=0.25):
        super().__init__()
        c_mid = max(8, int(c_in * mid_ratio))
        self.tap = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.tap(x)


class CLAG(nn.Module):
    """
    Cross-Layer Attention Guidance
    S' = (1 - α)*S + α*Up(Align(M))
    其中 α = Up(Tap(M_shallow)) ∈ [0,1]（按像素引导/门控）

    典型用法：用浅层注意力图引导 “上采样后的深层特征” 与目标层 S 融合。
    输入:
        x[0] = S_target:   (B, C_t, H, W)     # 目标层（被引导）
        x[1] = M_deep:     (B, C_d, Hm, Wm)   # 来自更深层的特征（待上采样）
        x[2] = M_shallow:  (B, C_s, Hs, Ws)   # 浅层（只用于提取注意力）
    输出:
        S_out: (B, C_t, H, W)
    """
    def __init__(self, c_t, c_d, c_s, align_mode="bilinear"):
        super().__init__()
        self.align_mode = align_mode

        # 将深层特征对齐到目标通道数（不改变空间尺寸）
        self.align_deep = nn.Conv2d(c_d, c_t, kernel_size=1, bias=False)
        # 浅层注意力提取器（1ch）
        self.tap = TapAttention(c_s)

        # 轻量门控修正（可选）：生成一个逐通道比例 β (0~1)，增强稳定性
        self.beta = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_t, c_t, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, \
            "CLAG expects [S_target, M_deep, M_shallow]"
        S, Mdeep, Mshallow = x

        B, Ct, H, W = S.shape

        # 1) 深层特征通道对齐 + 上采样到目标分辨率
        Mdeep_aligned = self.align_deep(Mdeep)
        Mdeep_up = F.interpolate(Mdeep_aligned, size=(H, W), mode=self.align_mode, align_corners=False)

        # 2) 浅层注意力图 (1ch) 并上采样到目标分辨率
        att = self.tap(Mshallow)                    # (B,1,Hs,Ws)
        att_up = F.interpolate(att, size=(H, W), mode=self.align_mode, align_corners=False)  # α

        # 3) 逐通道门控比例 β（提升数值稳定）
        beta = self.beta(S)                         # (B,Ct,1,1)

        # 4) 引导融合：S' = (1-α)*S + α*(β ⊙ Mdeep_up)
        S_guided = (1.0 - att_up) * S + att_up * (beta * Mdeep_up)
        return S_guided
