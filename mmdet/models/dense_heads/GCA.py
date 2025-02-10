import torch.nn as nn
from einops import rearrange


class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        # return rearrange(x, "b h w c -> b c h w")
        return x

class DWConv(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
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
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = DWConv(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GCA(nn.Module):
    def __init__(self, dim, latent_ratio=8, num_heads=8, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.attn_dim = dim // num_heads
        self.latent_num = dim // latent_ratio
        self.scale = self.attn_dim ** -0.5
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.latent_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim),
            nn.InstanceNorm2d(self.latent_num),
            nn.GELU(),
            nn.Conv2d(dim, self.latent_num, kernel_size=1, bias=False),
        )
        self.latent_norm_img = nn.InstanceNorm2d(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.norm1 = LayerNormProxy(dim)
        self.cpe1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim),
            nn.GELU()
        )

        self.latent_norm = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cpe2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim),
            nn.GELU()
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=nn.GELU)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x + self.cpe1(x)

        reshaped_x = rearrange(x, "b c h w -> b (h w) c")
        img = self.norm1(reshaped_x)

        latent_img = reshaped_x * ((H * W) ** -0.5)
        latent_x = self.latent_proj(x).flatten(2)

        latent_x = latent_x @ latent_img  # b, n, c
        latent_x = self.latent_norm(latent_x)

        q = self.q(img)
        kv = self.kv(latent_x)

        q = rearrange(q, "b n (nh c) -> b nh n c", nh=self.num_heads) * self.scale
        kv = rearrange(kv, "b n (d nh c) -> d b nh n c", d=2, nh=self.num_heads)
        k, v = kv[0], kv[1]

        attn = q @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b nh n c -> b n (nh c)")

        out = self.proj(out) + img

        out = rearrange(out, "b (h w) c -> b c h w", h=H)
        out = self.cpe2(out)
        out = rearrange(out, "b c h w -> b (h w) c")

        out = out + self.mlp(self.norm2(out), H, W)
        out = rearrange(out, "b (h w) c -> b c h w", h=H)
        return out
