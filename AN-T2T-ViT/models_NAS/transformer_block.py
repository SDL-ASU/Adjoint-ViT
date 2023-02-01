import torch
import torch.nn as nn
import numpy as np
from pytorch_image_models.models.layers import DropPath
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits, gumbel_noise, temperature, hard=False):
    y = logits + gumbel_noise
    y = F.softmax(y / temperature, dim=-1)
    if not hard:
       return y
    else:
      idx = torch.argmax(y)
      y_hard = torch.zeros_like(y).cuda()
      y_hard.scatter_(0, idx, 1)
      return y_hard

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., alpha=[1,2,3], depth=0, block_depth=2, total_depth=14):
        super().__init__()
        out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.depth = depth
        self.block_depth = block_depth
        self.total_depth = total_depth - 1
        self.alpha = alpha

    def forward(self, x, gumbel_weights, latency=0, prev_g_weights=0):
        l, N, C = x.shape
        x_large, x_small = x[:l//2], x[l//2:]
        x_large = self.fc1(x_large)
        x_large = self.act(x_large)
        x_large = self.drop(x_large)
        x_large = self.fc2(x_large)
        x_large = self.drop(x_large)

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            l = x_small.shape[2]

            if gumbel_weights[0] != 0:
                x_small_1 = torch.clone(x_small)
                x_small_1[:, :, l//self.alpha[0]:] = 0
                x_small_1 = self.fc1(x_small_1)
                x_small_1 = self.act(x_small_1)
                x_small_1 = self.drop(x_small_1)
                x_small_1 = self.fc2(x_small_1)
                l = x_small_1.shape[2]
                x_small_1[:, :, l//self.alpha[0]:] = 0
                x_small_1 = self.drop(x_small_1)
                x_small_1 *= gumbel_weights[0]
                flops_small_1 = 2*N*l//self.alpha[0]*self.hidden_features
                latency += flops_small_1*gumbel_weights[0]
            else:
                x_small_1 = 0
            
            if gumbel_weights[1] != 0:
                x_small_2 = torch.clone(x_small)
                x_small_2[:, :, l//self.alpha[1]:] = 0
                x_small_2 = self.fc1(x_small_2)
                x_small_2 = self.act(x_small_2)
                x_small_2 = self.drop(x_small_2)
                x_small_2 = self.fc2(x_small_2)
                l = x_small_2.shape[2]
                x_small_2[:, :, l//self.alpha[1]:] = 0
                x_small_2 = self.drop(x_small_2)
                x_small_2 *= gumbel_weights[1]
                flops_small_2 = 4*N*l//self.alpha[1]*self.hidden_features
                latency += flops_small_2*gumbel_weights[1]
            else:
                x_small_2 = 0
            
            if gumbel_weights[2] != 0:
                x_small_3 = torch.clone(x_small)
                x_small_3[:, :, l//self.alpha[2]:] = 0
                x_small_3 = self.fc1(x_small_3)
                x_small_3 = self.act(x_small_3)
                x_small_3 = self.drop(x_small_3)
                x_small_3 = self.fc2(x_small_3)
                l = x_small_3.shape[2]
                x_small_3[:, :, l//self.alpha[2]:] = 0
                x_small_3 = self.drop(x_small_3)
                x_small_3 *= gumbel_weights[2]
                flops_small_3 = 4*N*l//self.alpha[2]*self.hidden_features
                latency += flops_small_3*gumbel_weights[2]
            else:
                x_small_3 = 0

            concatinated_tensor = torch.cat((x_large, x_small_1 + x_small_2 + x_small_3), dim=0)
        else:
            concatinated_tensor = torch.cat((x_large, x_large), dim=0)

        return concatinated_tensor, latency, gumbel_weights


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., alpha=[1,2,3], depth=0, block_depth=3, total_depth=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.alpha = alpha
        self.depth = depth
        self.block_depth = block_depth
        self.total_depth = total_depth - 1

    def forward(self, x, gumbel_weights, latency = 0, prev_g_weights = 0):
        B, N, C = x.shape
        qkv_large = self.qkv(x[:B//2]).reshape(B//2, N, 3, self.num_heads, C //
                                               self.num_heads).permute(2, 0, 3, 1, 4)

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

            if gumbel_weights[0] != 0:
                qkv_small_1 = torch.clone(qkv_small)
                qkv_small_1[:, :, self.num_heads//self.alpha[0]:] = 0

                q_small_1, k_small_1, v_small_1 = qkv_small_1[0], qkv_small_1[1], qkv_small_1[2]

                attn_small_1 = (q_small_1 @ k_small_1.transpose(-2, -1)) * self.scale

                attn_small_1 = attn_small_1.softmax(dim=-1)

                attn_small_1 = self.attn_drop(attn_small_1)

                x_small_1 = (attn_small_1 @ v_small_1).transpose(1,2).reshape(B//2, N, C)

                flops_small_1 = 4 * N * N * C//self.num_heads * self.num_heads//self.alpha[0] + 2 * N * C//self.alpha[0] * C//self.alpha[0]

                x_small_1 = self.proj(x_small_1)
                l = x_small_1.shape[2]
                x_small_1[:, :, l//self.alpha[0]:] = 0
                x_small_1 = self.proj_drop(x_small_1)
                x_small_1 *= gumbel_weights[0]
                flops_small_1 *= gumbel_weights[0]
                latency += flops_small_1
            else:
                x_small_1 = 0
            
            if gumbel_weights[1] != 0:
                qkv_small_2 = torch.clone(qkv_small)
                qkv_small_2[:, :, self.num_heads//self.alpha[1]:] = 0

                q_small_2, k_small_2, v_small_2 = qkv_small_2[0], qkv_small_2[1], qkv_small_2[2]

                attn_small_2 = (q_small_2 @ k_small_2.transpose(-2, -1)) * self.scale

                attn_small_2 = attn_small_2.softmax(dim=-1)

                attn_small_2 = self.attn_drop(attn_small_2)

                x_small_2 = (attn_small_2 @ v_small_2).transpose(1,
                                                        2).reshape(B//2, N, C)
                
                flops_small_2 = 4 * N * N * C//self.num_heads * self.num_heads//self.alpha[1] + 2 * N * C//self.alpha[1] * C//self.alpha[1]

                x_small_2 = self.proj(x_small_2)
                l = x_small_2.shape[2]
                x_small_2[:, :, l//self.alpha[1]:] = 0
                x_small_2 = self.proj_drop(x_small_2)
                x_small_2 *= gumbel_weights[1]
                flops_small_2 *= gumbel_weights[1]
                latency += flops_small_2
            else:
                x_small_2 = 0
            
            if gumbel_weights[2] != 0:
                qkv_small_3 = torch.clone(qkv_small)
                qkv_small_3[:, :, self.num_heads//self.alpha[2]:] = 0

                q_small_3, k_small_3, v_small_3 = qkv_small_3[0], qkv_small_3[1], qkv_small_3[2]

                attn_small_3 = (q_small_3 @ k_small_3.transpose(-2, -1)) * self.scale

                attn_small_3 = attn_small_3.softmax(dim=-1)

                attn_small_3 = self.attn_drop(attn_small_3)

                x_small_3 = (attn_small_3 @ v_small_3).transpose(1,
                                                        2).reshape(B//2, N, C)
                
                flops_small_3 = 4 * N * N * C//self.num_heads * self.num_heads//self.alpha[2] + 2 * N * C//self.alpha[2] * C//self.alpha[2]

                x_small_3 = self.proj(x_small_3)
                l = x_small_3.shape[2]
                x_small_3[:, :, l//self.alpha[2]:] = 0
                x_small_3 = self.proj_drop(x_small_3)
                x_small_3 *= gumbel_weights[2]
                flops_small_3 *= gumbel_weights[2]
                latency += flops_small_3
            else:
                x_small_3 = 0
            
            concatinated_tensor = torch.cat((x_large, x_small_1 + x_small_2 + x_small_3), dim=0)
        else:
            concatinated_tensor = torch.cat((x_large, x_large), dim=0)
        return concatinated_tensor, latency, gumbel_weights


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0, alpha=[1,2,3], block_depth=3, total_depth=14):
        super().__init__()
        self.norm1_large = norm_layer(dim)
        self.norm1_small = norm_layer(dim)
        self.alpha = alpha
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                     qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, alpha=alpha, depth=depth, block_depth=block_depth, total_depth=total_depth)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_large = norm_layer(dim)
        self.norm2_small = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=act_layer, drop=drop, alpha=alpha, depth=depth, block_depth=block_depth, total_depth=total_depth)
        self.depth = depth
        self.block_depth = block_depth
        self.total_depth = total_depth - 1
        self.gumbel_weight = nn.Parameter(torch.rand(3))
        self.gumbel_noise = nn.Parameter(sample_gumbel(self.gumbel_weight.size()))
        self.gumbel_noise.requires_grad = False

    def forward(self, x, epoch = None, latency = 0, prev_g_weights = 0):
        l, _, dim = x.shape

        norm1_large = self.norm1_large(x[:l//2])

        g_weights = gumbel_softmax(self.gumbel_weight, self.gumbel_noise, 15*((0.956)**epoch), False)

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            norm1_small = self.norm1_small(x[l//2:])

            if g_weights[0] != 0:
                norm1_small_1 = torch.clone(norm1_small)
                norm1_small_1[:,:,dim//self.alpha[0]:] = 0
                norm1_small_1 *= g_weights[0]
            else:
                norm1_small_1 = 0
            
            if g_weights[1] != 0:
                norm1_small_2 = torch.clone(norm1_small)
                norm1_small_2[:,:,dim//self.alpha[1]:] = 0
                norm1_small_2 *= g_weights[1]
            else:
                norm1_small_2 = 0
            
            if g_weights[2] != 0:
                norm1_small_3 = torch.clone(norm1_small)
                norm1_small_3[:,:,dim//self.alpha[2]:] = 0
                norm1_small_3 *= g_weights[2]
            else:
                norm1_small_3 = 0

            norm1 = torch.cat((norm1_large, norm1_small_1 + norm1_small_2 + norm1_small_3), dim=0)
        else:
            norm1 = torch.cat((norm1_large, norm1_large), dim=0)

        attn, latency, prev_g_weights = self.attn(norm1, g_weights, latency, prev_g_weights)
        x = x + self.drop_path(attn)

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            if g_weights[0] != 0:
                x_DAN_1_1 = torch.clone(x[l//2:])
                x_DAN_1_1[l//2:, :, dim//self.alpha[0]:] = 0
                x_DAN_1_1 *= g_weights[0]
            else:
                x_DAN_1_1 = 0
            
            if g_weights[1] != 0:
                x_DAN_1_2 = torch.clone(x[l//2:])
                x_DAN_1_2[l//2:, :, dim//self.alpha[1]:] = 0
                x_DAN_1_2 *= g_weights[1]
            else:
                x_DAN_1_2 = 0
            
            if g_weights[2] != 0:
                x_DAN_1_3 = torch.clone(x[l//2:])
                x_DAN_1_3[l//2:, :, dim//self.alpha[2]:] = 0
                x_DAN_1_3 *= g_weights[2]
            else:
                x_DAN_1_3 = 0

            x[l//2:] = x_DAN_1_1 + x_DAN_1_2 + x_DAN_1_3
    
        norm2_large = self.norm2_large(x[:l//2])

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            norm2_small = self.norm2_small(x[l//2:])

            if g_weights[0] != 0:
                norm2_small_1 = torch.clone(norm2_small)
                norm2_small_1[:,:,dim//self.alpha[0]:] = 0
                norm2_small_1 *= g_weights[0]
            else:
                norm2_small_1 = 0
            
            if g_weights[1] != 0:
                norm2_small_2 = torch.clone(norm2_small)
                norm2_small_2[:,:,dim//self.alpha[1]:] = 0
                norm2_small_2 *= g_weights[1]
            else:
                norm2_small_2 = 0
            
            if g_weights[2] != 0:
                norm2_small_3 = torch.clone(norm2_small)
                norm2_small_3[:,:,dim//self.alpha[2]:] = 0
                norm2_small_3 *= g_weights[2]
            else:
                norm2_small_3 = 0

            norm2 = torch.cat((norm2_large, norm2_small_1 + norm2_small_2 + norm2_small_3), dim=0)
        else:
            norm2 = torch.cat((norm2_large, norm2_large), dim=0)

        mlp, latency, gumbel_weights = self.mlp(norm2, g_weights, latency, prev_g_weights)
        x = x + self.drop_path(mlp)

        if self.depth >= self.block_depth and self.depth <= self.total_depth - self.block_depth:
            if g_weights[0] != 0:
                x_DAN_2_1 = torch.clone(x[l//2:])
                x_DAN_2_1[l//2:, :, dim//self.alpha[0]:] = 0
                x_DAN_2_1 *= g_weights[0]
            else:
                x_DAN_2_1 = 0
            
            if g_weights[1] != 0:
                x_DAN_2_2 = torch.clone(x[l//2:])
                x_DAN_2_2[l//2:, :, dim//self.alpha[1]:] = 0
                x_DAN_2_2 *= g_weights[1]
            else:
                x_DAN_2_2 = 0
            
            if g_weights[2] != 0:
                x_DAN_2_3 = torch.clone(x[l//2:])
                x_DAN_2_3[l//2:, :, dim//self.alpha[2]:] = 0
                x_DAN_2_3 *= g_weights[2]
            else:
                x_DAN_2_3 = 0

            x[l//2:] = x_DAN_2_1 + x_DAN_2_2 + x_DAN_2_3

        return x, latency, g_weights


def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
