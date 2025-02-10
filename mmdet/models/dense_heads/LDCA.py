import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dwconv = DWConv(hidden_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        # x = self.act(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return rearrange(x, "b h w c -> b c h w")


class LDCA(nn.Module):
    def __init__(self, dim, ks=3, dim_reduction=4, rpb=True, mlp_ratio=4, n_groups=2, range_factor=9.,
                 channel_wise=False):
        super(LDCA, self).__init__()
        self.ks = ks
        self.dim = dim
        self.rpb = rpb
        self.range_factor = range_factor
        self.hidden_dim = dim // dim_reduction
        self.scale = self.hidden_dim ** -0.5

        self.norm11 = LayerNormProxy(dim)
        self.norm21 = LayerNormProxy(dim)
        self.norm12 = nn.LayerNorm(dim)
        self.qkv1 = nn.Conv2d(dim, dim * 3 // dim_reduction, 1, bias=False)
        self.qkv2 = nn.Conv2d(dim, dim * 2 // dim_reduction, 1, bias=False)

        self.n_groups = n_groups
        self.group_channel = self.hidden_dim // n_groups
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_channel, self.group_channel, 3, 1, 1,
                      groups=self.group_channel, bias=False),
            LayerNormProxy(self.group_channel),
            nn.GELU(),
            nn.Conv2d(self.group_channel, 2 * ks ** 2, 3, 1, 1)
        )
        self.conv_offset2 = nn.Sequential(
            nn.Conv2d(self.group_channel, self.group_channel, 3, 1, 1,
                      groups=self.group_channel, bias=False),
            LayerNormProxy(self.group_channel),
            nn.GELU(),
            nn.Conv2d(self.group_channel, 2 * ks ** 2, 3, 1, 1)
        )
        self.proj = nn.Linear(dim // dim_reduction, dim)

        pad = int((ks - 1) / 2)
        base = np.arange(-pad, pad + 1).astype(np.float32)
        base_y = np.repeat(base, ks)
        base_x = np.tile(base, ks)
        base_offset = np.stack([base_y, base_x], axis=1).flatten()
        base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)
        self.register_buffer("base_offset", base_offset, persistent=False)

        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, 1, self.ks * self.ks, dim // dim_reduction))
            trunc_normal_(self.relative_position_bias_table, std=.02)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=nn.GELU)

    def extract_feats(self, x, index_tensor, offset, ks=3):
        B, C, H, W = x.shape
        offset = rearrange(offset, "b (kh kw d) h w -> b kh h kw w d", kh=ks, kw=ks)
        offset = offset + index_tensor.view(1, 1, H, 1, W, 2)
        offset = offset.contiguous().view(B, ks * H, ks * W, 2)

        offset[..., 0] = (2 * offset[..., 0] / (H - 1) - 1)
        offset[..., 1] = (2 * offset[..., 1] / (W - 1) - 1)
        offset = offset.flip(-1)

        out = nn.functional.grid_sample(x, offset.to(x.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)
        out = rearrange(out, "b c (ksh h) (ksw w) -> b (ksh ksw) c h w", ksh=ks, ksw=ks)
        return out

    def forward(self, x, y):
        B, C, H, W = x.shape
        device = x.device
        x_, y_ = self.norm11(x), self.norm21(y)
        qkv1 = self.qkv1(x_)  # B, C, H, W
        kv2 = self.qkv2(y_)  # B, 2C, H, W

        q1 = qkv1[:, :self.hidden_dim, ...]
        kv1 = qkv1[:, self.hidden_dim:, ...]

        q1 = q1.reshape(B * self.n_groups, -1, H, W)
        kv1 = rearrange(kv1, "b (d g c) h w -> (b g) (d c) h w", d=2, g=self.n_groups)
        kv2 = rearrange(kv2, "b (d g c) h w -> (b g) (d c) h w", d=2, g=self.n_groups)
        pred_offset = self.conv_offset(q1).tanh().mul(self.range_factor) + self.base_offset.to(x.dtype)
        pred_offset2 = self.conv_offset2(q1).tanh().mul(self.range_factor) + self.base_offset.to(x.dtype)

        row_indices = torch.arange(H, device=device)
        col_indices = torch.arange(W, device=device)
        row_indices, col_indices = torch.meshgrid(row_indices, col_indices)
        index_tensor = torch.stack((row_indices, col_indices), dim=-1).view(1, H, W, 2)

        kv1 = self.extract_feats(kv1, index_tensor, pred_offset, self.ks)
        kv2 = self.extract_feats(kv2, index_tensor, pred_offset2, self.ks)

        kv1 = rearrange(kv1, "(b g) n (d c) h w -> d b (h w) n (g c)", g=self.n_groups, d=2)
        kv2 = rearrange(kv2, "(b g) n (d c) h w -> d b (h w) n (g c)", g=self.n_groups, d=2)

        k1, v1 = kv1[0], kv1[1]
        k2, v2 = kv2[0], kv2[1]

        if self.rpb:
            k1 = k1 + self.relative_position_bias_table
            k2 = k2 + self.relative_position_bias_table

        q = rearrange(q1, "(b g) c h w -> b (h w) () (g c)", g=self.n_groups) * self.scale
        k = torch.cat([k1, k2], 2)
        v = torch.cat([v1, v2], 2)
        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).squeeze(-2)
        out = self.proj(out) + rearrange(x, "b c h w -> b (h w) c")

        out = out + self.mlp(self.norm12(out), H, W)
        out = rearrange(out, "b (h w) c -> b c h w", h=H)
        return out