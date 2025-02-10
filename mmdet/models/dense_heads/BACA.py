import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return rearrange(x, "b h w c -> b c h w")


class BACA(nn.Module):
    def __init__(self, dim, num_class=80, num_anchor=1, ks=3, num_heads=4, dim_reduction=4, group=2, rpb=True,
                 mul_factor=5, dynamic_mul=False):
        super().__init__()
        self.dim = dim
        self.ks = ks
        self.rpb = rpb
        self.group = group
        self.num_heads = num_heads
        self.mul_factor = mul_factor
        self.dynamic_mul = dynamic_mul
        head_dim = dim // num_heads // dim_reduction
        self.scale = head_dim ** -0.5
        self.dim_reduction = dim_reduction
        self.num_anchor = num_anchor

        self.norm1 = LayerNormProxy(dim)

        self.qkv = nn.Conv2d(dim, dim * 3 // self.dim_reduction, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim // dim_reduction, dim)
        self.cls_conv = nn.Linear(dim, num_class)

        self.group_channel = dim // dim_reduction // self.group

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_channel, self.group_channel, 3, 1, 1,
                      groups=self.group_channel, bias=False),
            LayerNormProxy(self.group_channel),
            nn.GELU(),
            nn.Conv2d(self.group_channel, self.num_anchor * 2 * ks ** 2, 3, 1, 1)
        )

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.num_heads, 1, 1, self.ks * self.ks, dim // self.dim_reduction // self.num_heads))
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def extract_feats(self, x, offset, ks=3):
        B, C, H, W = x.shape
        offset = rearrange(offset, "b (na kh kw d) h w -> b (na kh h) (kw w) d", na=self.num_anchor, kh=ks, kw=ks)
        offset[..., 0] = (2 * offset[..., 0] / (H - 1) - 1)
        offset[..., 1] = (2 * offset[..., 1] / (W - 1) - 1)
        offset = offset.flip(-1)
        out = nn.functional.grid_sample(x, offset.to(x.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)
        out = rearrange(out, "b c (na ksh h) (ksw w) -> b (na ksh ksw) c h w", na=self.num_anchor, ksh=ks, ksw=ks)
        return out

    def forward(self, x, offset):
        # x -> B, C, H, W
        # offset -> B, N*9*2, H, W
        B, C, H, W = x.shape
        x_ = self.norm1(x)

        qkv = self.qkv(x_).reshape(B, 3, C // self.dim_reduction, H, W).transpose(0, 1)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, C, H, W

        # q_off = self.extract_feats(q, offset, self.ks)
        # q_off = rearrange(q_off, "b (na ksh ksw) (g c) h w -> (b na h w g) c ksh ksw", g=self.group, na=self.num_anchor, ksh=self.ks, ksw=self.ks)
        # pred_offset = self.conv_offset(q_off)
        # pred_offset = rearrange(pred_offset, "(b na h w g) c ksh ksw -> (b g) (na ksh ksw c) h w", b=B, na=self.num_anchor, h=H, w=W, g=self.group, )

        q_off = q.reshape(B * self.group, -1, H, W)
        pred_offset = self.conv_offset(q_off)
        if self.dynamic_mul:
            mul_offset_base = offset.reshape(B, 1, self.num_anchor, -1, H, W).detach().clone()
            mul_offset = offset.new_zeros(B, 1, self.num_anchor, 2, H, W)
            mul_offset[..., 0, :, :] = (mul_offset_base[..., -2, :, :] - mul_offset_base[..., 0, :, :])
            mul_offset[..., 1, :, :] = (mul_offset_base[..., -1, :, :] - mul_offset_base[..., 1, :, :])
            mul_offset = mul_offset.repeat(1, self.group, 1, self.ks * self.ks, 1, 1).reshape(B * self.group, -1, H, W)

            offset = offset.reshape(B, 1, -1, H, W).repeat(1, self.group, 1, 1, 1).reshape(B * self.group, -1, H, W)
            offset = pred_offset.tanh().mul(mul_offset) + offset
        else:
            offset = offset.reshape(B, 1, -1, H, W).repeat(1, self.group, 1, 1, 1).reshape(B * self.group, -1, H, W)
            offset = pred_offset.tanh().mul(self.mul_factor) + offset

        k = k.reshape(B * self.group, -1, H, W)
        v = v.reshape(B * self.group, -1, H, W)
        k = self.extract_feats(k, offset, self.ks)  # B, N, C, H, W
        v = self.extract_feats(v, offset, self.ks)  # B, N, C, H, W

        q = rearrange(q, "b (nh c) h w -> b nh (h w) () () c", nh=self.num_heads)
        q = q.repeat(1, 1, 1, self.num_anchor, 1, 1)  # B, Nh, H*W, N, 1, C

        k = rearrange(k, "(b g) (na n) c h w -> b (h w) na n (g c)", na=self.num_anchor, g=self.group)
        v = rearrange(v, "(b g) (na n) c h w -> b (h w) na n (g c)", na=self.num_anchor, g=self.group)
        k = rearrange(k, "b n1 na n (nh c) -> b nh n1 na n c", na=self.num_anchor, nh=self.num_heads)
        v = rearrange(v, "b n1 na n (nh c) -> b nh n1 na n c", na=self.num_anchor, nh=self.num_heads)

        if self.rpb:
            k = k + self.relative_position_bias_table

        q = q * self.scale
        attn = q @ k.transpose(-1, -2)
        attn = self.softmax(attn)
        out = (attn @ v).squeeze(-2)  # B, nh, H*W, C // nh
        out = rearrange(out, "b nh (h w) na c -> b (h w) na (nh c)", nh=self.num_heads, h=H)
        out = self.proj(out) + rearrange(x, "b c h w -> b (h w) () c")

        cls_pred = self.cls_conv(out)
        cls_pred = rearrange(cls_pred, "b (h w) na c -> b (na c) h w", h=H)
        return cls_pred