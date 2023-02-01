"""
Borrow from timm(https://github.com/rwightman/pytorch-image-models)
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import DropPath

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., depth=0, compression_factor=2, block_depth=2, total_depth=14):
        super().__init__()
        self.depth = depth
        self.compression_factor = compression_factor
        self.block_depth = block_depth
        self.total_depth = total_depth - 1

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        l, _, _ = x.shape
        x_large, x_small = x[:l//2], x[l//2:]
        x_large = self.fc1(x_large)
        x_large = self.act(x_large)
        x_large = self.drop(x_large)
        x_large = self.fc2(x_large)
        x_large = self.drop(x_large)

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            x_small = self.fc1(x_small)
            l = x_small.shape[2]
            x_small[:, :, l//self.compression_factor:] = 0
            x_small = self.act(x_small)
            x_small = self.drop(x_small)
            x_small = self.fc2(x_small)
            l = x_small.shape[2]
            x_small[:, :, l//self.compression_factor:] = 0
            x_small = self.drop(x_small)

            concatinated_tensor = torch.cat((x_large, x_small), dim=0)
        else:
            concatinated_tensor = torch.cat((x_large, x_large), dim=0)

        return concatinated_tensor

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., depth=0, compression_factor=2, block_depth=2, total_depth=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.depth = depth
        self.compression_factor = compression_factor
        self.block_depth = block_depth
        self.total_depth = total_depth - 1

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_large = self.qkv(x[:B//2]).reshape(B//2, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_large, k_large, v_large = qkv_large[0], qkv_large[1], qkv_large[2]

        attn_large = (q_large @ k_large.transpose(-2, -1)) * self.scale
        attn_large = attn_large.softmax(dim=-1)
        attn_large = self.attn_drop(attn_large)

        x_large = (attn_large @ v_large).transpose(1, 2).reshape(B//2, N, C)

        x_large = self.proj(x_large)
        x_large = self.proj_drop(x_large)

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            qkv_small = self.qkv(x[B//2:]).reshape(B//2, N, 3, self.num_heads, C //
                                                   self.num_heads).permute(2, 0, 3, 1, 4)

            qkv_small[:, :, self.num_heads//self.compression_factor:] = 0

            q_small, k_small, v_small = qkv_small[0], qkv_small[1], qkv_small[2]

            attn_small = (q_small @ k_small.transpose(-2, -1)) * self.scale
            attn_small = attn_small.softmax(dim=-1)
            attn_small = self.attn_drop(attn_small)

            x_small = (attn_small @ v_small).transpose(1,2).reshape(B//2, N, C)
            x_small = self.proj(x_small)
            l = x_small.shape[2]
            x_small[:, :, l//self.compression_factor:] = 0
            x_small = self.proj_drop(x_small)

            concatinated_tensor = torch.cat((x_large, x_small), dim=0)
        else:
            concatinated_tensor = torch.cat((x_large, x_large), dim=0)
        return concatinated_tensor

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0, compression_factor=2, block_depth=2, total_depth=14):
        super().__init__()
        self.depth = depth
        self.compression_factor = compression_factor
        self.block_depth = block_depth
        self.total_depth = total_depth - 1
        self.norm1_large = norm_layer(dim)
        self.norm1_small = norm_layer(dim)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, depth=depth, compression_factor=compression_factor, block_depth=block_depth, total_depth=total_depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_large = norm_layer(dim)
        self.norm2_small = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, depth=depth, compression_factor=compression_factor, block_depth=block_depth, total_depth=total_depth)

    def forward(self, x):
        l, _, dim = x.shape

        norm1_large = self.norm1_large(x[:l//2])

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            norm1_small = self.norm1_small(x[l//2:])
            norm1_small[:,:,dim//self.compression_factor:] = 0
            norm1 = torch.cat((norm1_large, norm1_small), dim=0)
        else:
            norm1 = torch.cat((norm1_large, norm1_large), dim=0)

        x = x + self.drop_path(self.attn(norm1))

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            x[l//2:, :, dim//self.compression_factor:] = 0

        norm2_large = self.norm2_large(x[:l//2])

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            norm2_small = self.norm2_small(x[l//2:])
            norm2_small[:,:,dim//self.compression_factor:] = 0
            norm2 = torch.cat((norm2_large, norm2_small), dim=0)
        else:
            norm2 = torch.cat((norm2_large, norm2_large), dim=0)

        x = x + self.drop_path(self.mlp(norm2))

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            x[l//2:, :, dim//self.compression_factor:] = 0

        return x


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
