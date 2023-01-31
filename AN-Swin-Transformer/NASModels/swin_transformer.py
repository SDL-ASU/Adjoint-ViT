import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    import os, sys

    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax(logits, gumbel_noise, temperature, hard=False):
    y = logits + gumbel_noise
    y = nn.functional.softmax(y / temperature, dim=-1)
    if not hard:
       return y
    else:
      idx = torch.argmax(y)
      y_hard = torch.zeros_like(y).cuda()
      y_hard.scatter_(0, idx, 1)
      return y_hard

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., alpha = [1, 2, 4, 8], stage = 0, depth = 0, num_heads = 4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.alpha = alpha
        self.stage = stage
        self.depth = depth
        self.num_heads = num_heads

    def forward(self, x, H, W, mlp_ratio, gumbel_weights, latency=0):
        l, _, _ = x.shape
        x_large, x_small = x[:l//2], x[l//2:]
        x_large = self.fc1(x_large)
        x_large = self.act(x_large)
        x_large = self.drop(x_large)
        x_large = self.fc2(x_large)
        x_large = self.drop(x_large)

        if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
            l = x_small.shape[2]

            if gumbel_weights[0] != 0 and self.alpha[0] <= self.num_heads:
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
                flops = 2 * H * W * (self.in_features // self.alpha[0]) * (self.in_features // self.alpha[0]) * mlp_ratio
                latency += flops * gumbel_weights[0]
            else:
                x_small_1 = 0
            
            if gumbel_weights[1] != 0 and self.alpha[1] <= self.num_heads:
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
                flops = 2 * H * W * (self.in_features // self.alpha[1]) * (self.in_features // self.alpha[1]) * mlp_ratio
                latency += flops * gumbel_weights[1]
            else:
                x_small_2 = 0
            
            if gumbel_weights[2] != 0 and self.alpha[2] <= self.num_heads:
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
                flops = 2 * H * W * (self.in_features // self.alpha[2]) * (self.in_features // self.alpha[2]) * mlp_ratio
                latency += flops * gumbel_weights[2]
            else:
                x_small_3 = 0
            
            if gumbel_weights[3] != 0 and self.alpha[3] <= self.num_heads:
                x_small_4 = torch.clone(x_small)
                x_small_4[:, :, l//self.alpha[3]:] = 0
                x_small_4 = self.fc1(x_small_4)
                x_small_4 = self.act(x_small_4)
                x_small_4 = self.drop(x_small_4)
                x_small_4 = self.fc2(x_small_4)
                l = x_small_4.shape[2]
                x_small_4[:, :, l//self.alpha[3]:] = 0
                x_small_4 = self.drop(x_small_4)
                x_small_4 *= gumbel_weights[3]
                flops = 2 * H * W * (self.in_features // self.alpha[3]) * (self.in_features // self.alpha[3]) * mlp_ratio
                latency += flops * gumbel_weights[3]
            else:
                x_small_4 = 0

            concatinated_tensor = torch.cat((x_large, x_small_1 + x_small_2 + x_small_3 + x_small_4), dim=0)
        else:
            concatinated_tensor = torch.cat((x_large, x_large), dim=0)
        
        return concatinated_tensor, latency

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., alpha = [1, 2, 4, 8], stage = 0, depth = 0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index_large = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index_small = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index_large", relative_position_index_large)
        self.register_buffer("relative_position_index_small", relative_position_index_small)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.alpha = alpha
        self.stage = stage
        self.depth = depth

    def forward(self, x, gumbel_weights, H, W, mask_large=None, mask_small=None, latency = 0):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        nW = H * W / self.window_size[0] / self.window_size[1] # for latency computation
        n = self.window_size[0] * self.window_size[1]

        B, N, C = x.shape
        qkv_large = self.qkv(x[:B // 2]).reshape(B // 2, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_large, k_large, v_large = qkv_large[0], qkv_large[1], qkv_large[2]  # make torchscript happy (cannot use tensor as tuple)

        q_large = q_large * self.scale
        attn_large = (q_large @ k_large.transpose(-2, -1))

        relative_position_bias_large = self.relative_position_bias_table[self.relative_position_index_large.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias_large = relative_position_bias_large.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn_large = attn_large + relative_position_bias_large.unsqueeze(0)

        if mask_large is not None:
            nW = mask_large.shape[0]
            attn_large = attn_large.view(B // (2 * nW), nW, self.num_heads, N, N) + mask_large.unsqueeze(1).unsqueeze(0)
            attn_large = attn_large.view(-1, self.num_heads, N, N)

        attn_large = self.softmax(attn_large)

        attn_large = self.attn_drop(attn_large)

        x_large = (attn_large @ v_large).transpose(1, 2).reshape(B // 2, N, C)
        x_large = self.proj(x_large)
        x_large = self.proj_drop(x_large)

        if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
            qkv_small = self.qkv(x[B // 2:]).reshape(B//2, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
            if gumbel_weights[0] != 0 and self.alpha[0] <= self.num_heads:
                qkv_small_1 = torch.clone(qkv_small)
                qkv_small_1[:, :, self.num_heads//self.alpha[0]:] = 0

                q_small_1, k_small_1, v_small_1 = qkv_small_1[0], qkv_small_1[1], qkv_small_1[2]  # make torchscript happy (cannot use tensor as tuple)

                q_small_1 = q_small_1 * self.scale
                attn_small_1 = (q_small_1 @ k_small_1.transpose(-2, -1))

                relative_position_bias_small_1 = self.relative_position_bias_table[self.relative_position_index_small.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias_small_1 = relative_position_bias_small_1.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn_small_1 = attn_small_1 + relative_position_bias_small_1.unsqueeze(0)

                if mask_small is not None:
                    nW = mask_small.shape[0]
                    attn_small_1 = attn_small_1.view(B // (2 * nW), nW, self.num_heads, N, N) + mask_small.unsqueeze(1).unsqueeze(0)
                    attn_small_1 = attn_small_1.view(-1, self.num_heads, N, N)

                attn_small_1 = self.softmax(attn_small_1)

                attn_small_1 = self.attn_drop(attn_small_1)

                x_small_1 = (attn_small_1 @ v_small_1).transpose(1, 2).reshape(B // 2, N, C)

                x_small_1 = self.proj(x_small_1)
                x_small_1[:, :, C//self.alpha[0]:] = 0
                x_small_1 = self.proj_drop(x_small_1)
                x_small_1 *= gumbel_weights[0]
                flops = 4 * n * (self.dim // self.alpha[0]) * (self.dim // self.alpha[0]) + 2 * (self.num_heads // self.alpha[0]) * n * n * (self.dim // self.num_heads)
                flops *= (nW // self.alpha[0])
                latency += flops * gumbel_weights[0]
            else:
                x_small_1 = 0
            
            if gumbel_weights[1] != 0 and self.alpha[1] <= self.num_heads:
                qkv_small_2 = torch.clone(qkv_small)
                qkv_small_2[:, :, self.num_heads//self.alpha[1]:] = 0

                q_small_2, k_small_2, v_small_2 = qkv_small_2[0], qkv_small_2[1], qkv_small_2[2]  # make torchscript happy (cannot use tensor as tuple)

                q_small_2 = q_small_2 * self.scale
                attn_small_2 = (q_small_2 @ k_small_2.transpose(-2, -1))

                relative_position_bias_small_2 = self.relative_position_bias_table[self.relative_position_index_small.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias_small_2 = relative_position_bias_small_2.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn_small_2 = attn_small_2 + relative_position_bias_small_2.unsqueeze(0)

                if mask_small is not None:
                    nW = mask_small.shape[0]
                    attn_small_2 = attn_small_2.view(B // (2 * nW), nW, self.num_heads, N, N) + mask_small.unsqueeze(1).unsqueeze(0)
                    attn_small_2 = attn_small_2.view(-1, self.num_heads, N, N)

                attn_small_2 = self.softmax(attn_small_2)

                attn_small_2 = self.attn_drop(attn_small_2)

                x_small_2 = (attn_small_2 @ v_small_2).transpose(1, 2).reshape(B // 2, N, C)
               
                x_small_2 = self.proj(x_small_2)
                x_small_2[:, :, C//self.alpha[1]:] = 0
                x_small_2 = self.proj_drop(x_small_2)
                x_small_2 *= gumbel_weights[1]
                flops = 4 * n * (self.dim // self.alpha[1]) * (self.dim // self.alpha[1]) + 2 * (self.num_heads // self.alpha[1]) * n * n * (self.dim // self.num_heads)
                flops *= (nW // self.alpha[1])
                latency += flops * gumbel_weights[1]
            else:
                x_small_2 = 0
            
            if gumbel_weights[2] != 0 and self.alpha[2] <= self.num_heads:
                qkv_small_3 = torch.clone(qkv_small)
                qkv_small_3[:, :, self.num_heads//self.alpha[2]:] = 0

                q_small_3, k_small_3, v_small_3 = qkv_small_3[0], qkv_small_3[1], qkv_small_3[2]  # make torchscript happy (cannot use tensor as tuple)

                q_small_3 = q_small_3 * self.scale
                attn_small_3 = (q_small_3 @ k_small_3.transpose(-2, -1))

                relative_position_bias_small_3 = self.relative_position_bias_table[self.relative_position_index_small.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias_small_3 = relative_position_bias_small_3.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn_small_3 = attn_small_3 + relative_position_bias_small_3.unsqueeze(0)

                if mask_small is not None:
                    nW = mask_small.shape[0]
                    attn_small_3 = attn_small_3.view(B // (2 * nW), nW, self.num_heads, N, N) + mask_small.unsqueeze(1).unsqueeze(0)
                    attn_small_3 = attn_small_3.view(-1, self.num_heads, N, N)

                attn_small_3 = self.softmax(attn_small_3)

                attn_small_3 = self.attn_drop(attn_small_3)

                x_small_3 = (attn_small_3 @ v_small_3).transpose(1, 2).reshape(B // 2, N, C)
                
                x_small_3 = self.proj(x_small_3)
                x_small_3[:, :, C//self.alpha[2]:] = 0
                x_small_3 = self.proj_drop(x_small_3)
                x_small_3 *= gumbel_weights[2]
                flops = 4 * n * (self.dim // self.alpha[2]) * (self.dim // self.alpha[2]) + 2 * (self.num_heads // self.alpha[2]) * n * n * (self.dim // self.num_heads)
                flops *= (nW // self.alpha[2])
                latency += flops * gumbel_weights[2]
            else:
                x_small_3 = 0
            
            if gumbel_weights[3] != 0 and self.alpha[3] <= self.num_heads:
                qkv_small_4 = torch.clone(qkv_small)
                qkv_small_4[:, :, self.num_heads//self.alpha[3]:] = 0

                q_small_4, k_small_4, v_small_4 = qkv_small_4[0], qkv_small_4[1], qkv_small_4[2]  # make torchscript happy (cannot use tensor as tuple)

                q_small_4 = q_small_4 * self.scale
                attn_small_4 = (q_small_4 @ k_small_4.transpose(-2, -1))

                relative_position_bias_small_4 = self.relative_position_bias_table[self.relative_position_index_small.view(-1)].view(
                    self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias_small_4 = relative_position_bias_small_4.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn_small_4 = attn_small_4 + relative_position_bias_small_4.unsqueeze(0)

                if mask_small is not None:
                    nW = mask_small.shape[0]
                    attn_small_4 = attn_small_4.view(B // (2 * nW), nW, self.num_heads, N, N) + mask_small.unsqueeze(1).unsqueeze(0)
                    attn_small_4 = attn_small_4.view(-1, self.num_heads, N, N)

                attn_small_4 = self.softmax(attn_small_4)

                attn_small_4 = self.attn_drop(attn_small_4)

                x_small_4 = (attn_small_4 @ v_small_4).transpose(1, 2).reshape(B // 2, N, C)
                
                x_small_4 = self.proj(x_small_4)
                x_small_4[:, :, C//self.alpha[3]:] = 0
                x_small_4 = self.proj_drop(x_small_4)
                x_small_4 *= gumbel_weights[3]
                flops = 4 * n * (self.dim // self.alpha[3]) * (self.dim // self.alpha[3]) + 2 * (self.num_heads // self.alpha[3]) * n * n * (self.dim // self.num_heads)
                flops *= (nW // self.alpha[3]) 
                latency += flops * gumbel_weights[3]
            else:
                x_small_4 = 0
            
            concatinated_tensor = torch.cat((x_large, x_small_1 + x_small_2 + x_small_3 + x_small_4), dim=0)
        else:
            concatinated_tensor = torch.cat((x_large, x_large), dim=0)
        return concatinated_tensor, latency

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=True, alpha = [1, 2, 4, 8], stage = 0, depth = 0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_large = norm_layer(dim)
        self.norm1_small = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, alpha=alpha, stage=stage, depth=depth)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_large = norm_layer(dim)
        self.norm2_small = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.alpha = alpha
        self.stage = stage
        self.depth = depth
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, alpha=alpha, stage=stage, depth=depth, num_heads=num_heads)

        self.gumbel_weight = nn.Parameter(torch.rand(4))
        self.gumbel_noise = nn.Parameter(sample_gumbel(self.gumbel_weight.size()), requires_grad = False)
    
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask_large = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask_large = attn_mask_large.masked_fill(attn_mask_large != 0, float(-100.0)).masked_fill(attn_mask_large == 0, float(0.0))
            attn_mask_small = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask_small = attn_mask_small.masked_fill(attn_mask_small != 0, float(-100.0)).masked_fill(attn_mask_small == 0, float(0.0))
        else:
            attn_mask_large = None
            attn_mask_small = None

        self.register_buffer("attn_mask_large", attn_mask_large)
        self.register_buffer("attn_mask_small", attn_mask_small)
        self.fused_window_process = fused_window_process

    def forward(self, x, epoch, latency = 0):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        gumbel_weights = gumbel_softmax(self.gumbel_weight, self.gumbel_noise, 15*((0.956)**epoch), False)

        shortcut = x
        x_large = self.norm1_large(x[:B//2])
        x_large = x_large.view(B//2, H, W, C)

        if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
            x_small = self.norm1_small(x[B//2:])
            if gumbel_weights[0] != 0 and self.alpha[0] <= self.num_heads:
                x_small_1 = torch.clone(x_small)
                x_small_1[:, :, C//self.alpha[0]:] = 0
                x_small_1 = x_small_1.view(B//2, H, W, C)
                x_small_1 *= gumbel_weights[0]
                flops = (self.dim // self.alpha[0]) * H * W
                latency += flops * gumbel_weights[0]
            else:
                x_small_1 = 0
            
            if gumbel_weights[1] != 0 and self.alpha[1] <= self.num_heads:
                x_small_2 = torch.clone(x_small)
                x_small_2[:, :, C//self.alpha[1]:] = 0
                x_small_2 = x_small_2.view(B//2, H, W, C)
                x_small_2 *= gumbel_weights[1]
                flops = (self.dim // self.alpha[1]) * H * W
                latency += flops * gumbel_weights[1]
            else:
                x_small_2 = 0
            
            if gumbel_weights[2] != 0 and self.alpha[2] <= self.num_heads:
                x_small_3 = torch.clone(x_small)
                x_small_3[:, :, C//self.alpha[2]:] = 0
                x_small_3 = x_small_3.view(B//2, H, W, C)
                x_small_3 *= gumbel_weights[2]
                flops = (self.dim // self.alpha[2]) * H * W
                latency += flops * gumbel_weights[2]
            else:
                x_small_3 = 0
            
            if gumbel_weights[3] != 0 and self.alpha[3] <= self.num_heads:
                x_small_4 = torch.clone(x_small)
                x_small_4[:, :, C//self.alpha[3]:] = 0
                x_small_4 = x_small_4.view(B//2, H, W, C)
                x_small_4 *= gumbel_weights[3]
                flops = (self.dim // self.alpha[3]) * H * W
                latency += flops * gumbel_weights[3]
            else:
                x_small_4 = 0

            x = torch.cat((x_large, x_small_1 + x_small_2 + x_small_3 + x_small_4), dim = 0)
        else:
            x = torch.cat((x_large, x_large), dim = 0)
        
        l = x.shape[0]
        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x_large = torch.roll(x[:l//2], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows_large = window_partition(shifted_x_large, self.window_size)  # nW*B, window_size, window_size, C

                if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
                    shifted_x_small = torch.roll(x[l//2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                    # partition windows
                    x_windows_small = window_partition(shifted_x_small, self.window_size)  # nW*B, window_size, window_size, C
                    x_windows = torch.cat((x_windows_large, x_windows_small), dim = 0)
                else:
                    x_windows = torch.cat((x_windows_large, x_windows_large), dim = 0)
            else:
                x_windows_large = WindowProcess.apply(x[:l//2], B//2, H, W, C, -self.shift_size, self.window_size)
                if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
                    x_windows_small = WindowProcess.apply(x[l//2:], B//2, H, W, C, -self.shift_size, self.window_size)
                    x_windows = torch.cat((x_windows_large, x_windows_small), dim = 0)
                else:
                    x_windows = torch.cat((x_windows_large, x_windows_large), dim = 0)
        else:
            shifted_x_large = x[:l//2]
            # partition windows
            x_windows_large = window_partition(shifted_x_large, self.window_size)  # nW*B, window_size, window_size, C
            if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
                shifted_x_small = x[l//2:]
                # partition windows
                x_windows_small = window_partition(shifted_x_small, self.window_size)  # nW*B, window_size, window_size, C
                x_windows = torch.cat((x_windows_large, x_windows_small), dim = 0)
            else:
                x_windows = torch.cat((x_windows_large, x_windows_large), dim = 0)
        
        l = x_windows.shape[0]
        x_windows_large = x_windows[:l//2]
        x_windows_large = x_windows_large.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
            x_windows_small = x_windows[l//2:]
            x_windows_small = x_windows_small.view(-1, self.window_size * self.window_size, C)
            x_windows = torch.cat((x_windows_large, x_windows_small), dim = 0)
        else:
            x_windows = torch.cat((x_windows_large, x_windows_large), dim = 0)

        # W-MSA/SW-MSA
        attn_windows, latency = self.attn(x_windows, gumbel_weights=gumbel_weights, H=H, W=W, mask_large=self.attn_mask_large, mask_small=self.attn_mask_small, latency=latency)  # nW*B, window_size*window_size, C

        l = attn_windows.shape[0]
        # merge windows
        attn_windows_large = attn_windows[:l//2].view(-1, self.window_size, self.window_size, C)
        if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
            attn_windows_small = attn_windows[l//2:].view(-1, self.window_size, self.window_size, C)
            attn_windows = torch.cat((attn_windows_large, attn_windows_small), dim = 0)
        else:
            attn_windows = torch.cat((attn_windows_large, attn_windows_large), dim = 0)
        
        l = attn_windows.shape[0]
        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x_large = window_reverse(attn_windows[:l//2], self.window_size, H, W)  # B H' W' C
                x_large = torch.roll(shifted_x_large, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
                    shifted_x_small = window_reverse(attn_windows[l//2:], self.window_size, H, W)  # B H' W' C
                    x_small = torch.roll(shifted_x_small, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                    x = torch.cat((x_large, x_small), dim = 0)
                else:
                    x = torch.cat((x_large, x_large), dim = 0)
            else:
                x_large = WindowProcessReverse.apply(attn_windows[:l//2], B//2, H, W, C, self.shift_size, self.window_size)
                if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
                    x_small = WindowProcessReverse.apply(attn_windows[l//2:], B//2, H, W, C, self.shift_size, self.window_size)
                    x = torch.cat((x_large, x_small), dim = 0)
                else:
                    x = torch.cat((x_large, x_large), dim = 0)
        else:
            shifted_x_large = window_reverse(attn_windows[:l//2], self.window_size, H, W)  # B H' W' C
            x_large = shifted_x_large
            if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
                shifted_x_small = window_reverse(attn_windows[l//2:], self.window_size, H, W)
                x_small = shifted_x_small
                x = torch.cat((x_large, x_small), dim = 0)
            else:
                x = torch.cat((x_large, x_large), dim = 0)
            
        x_large = x[:B//2].view(B//2, H * W, C)
        if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
            x_small = x[B//2:].view(B//2, H * W, C)
            x = torch.cat((x_large, x_small), dim = 0)
        else:
            x = torch.cat((x_large, x_large), dim = 0)

        x = shortcut + self.drop_path(x)
        shortcut = x
        # FFN
        x_large = self.norm2_large(x[:B//2])
        if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
            x_small = self.norm2_small(x[B//2:])
            if gumbel_weights[0] != 0 and self.alpha[0] <= self.num_heads:
                x_small_1 = torch.clone(x_small)
                x_small_1[:, :, C//self.alpha[0]:] = 0
                x_small_1 *= gumbel_weights[0]
            else:
                x_small_1 = 0
            
            if gumbel_weights[1] != 0 and self.alpha[1] <= self.num_heads:
                x_small_2 = torch.clone(x_small)
                x_small_2[:, :, C//self.alpha[1]:] = 0
                x_small_2 *= gumbel_weights[1]
            else:
                x_small_2 = 0
            
            if gumbel_weights[2] != 0 and self.alpha[2] <= self.num_heads:
                x_small_3 = torch.clone(x_small)
                x_small_3[:, :, C//self.alpha[2]:] = 0
                x_small_3 *= gumbel_weights[2]
            else:
                x_small_3 = 0
            
            if gumbel_weights[3] != 0 and self.alpha[3] <= self.num_heads:
                x_small_4 = torch.clone(x_small)
                x_small_4[:, :, C//self.alpha[3]:] = 0
                x_small_4 *= gumbel_weights[3]
            else:
                x_small_4 = 0

            x = torch.cat((x_large, x_small_1 + x_small_2 + x_small_3 + x_small_4), dim = 0)
        else:
            x = torch.cat((x_large, x_large), dim = 0)

        x, latency = self.mlp(x, H=H, W=W, mlp_ratio=self.mlp_ratio, gumbel_weights=gumbel_weights, latency=latency)
        x = shortcut + self.drop_path(x)

        if (0 <= self.stage < 3 and self.stage + self.depth >= 1) or (self.stage == 3 and self.depth < 1):
            x_small = x[B//2:]
            if gumbel_weights[0] != 0 and self.alpha[0] <= self.num_heads:
                x_small_1 = torch.clone(x_small)
                x_small_1[:, :, C//self.alpha[0]:] = 0
                x_small_1 *= gumbel_weights[0]
                flops = (self.dim // self.alpha[0]) * H * W
                latency += flops * gumbel_weights[0]
            else:
                x_small_1 = 0
            
            if gumbel_weights[1] != 0 and self.alpha[1] <= self.num_heads:
                x_small_2 = torch.clone(x_small)
                x_small_2[:, :, C//self.alpha[1]:] = 0
                x_small_2 *= gumbel_weights[1]
                flops = (self.dim // self.alpha[1]) * H * W
                latency += flops * gumbel_weights[1]
            else:
                x_small_2 = 0
            
            if gumbel_weights[2] != 0 and self.alpha[2] <= self.num_heads:
                x_small_3 = torch.clone(x_small)
                x_small_3[:, :, C//self.alpha[2]:] = 0
                x_small_3 *= gumbel_weights[2]
                flops = (self.dim // self.alpha[2]) * H * W
                latency += flops * gumbel_weights[2]
            else:
                x_small_3 = 0
            
            if gumbel_weights[3] != 0 and self.alpha[3] <= self.num_heads:
                x_small_4 = torch.clone(x_small)
                x_small_4[:, :, C//self.alpha[3]:] = 0
                x_small_4 *= gumbel_weights[3]
                flops = (self.dim // self.alpha[3]) * H * W
                latency += flops * gumbel_weights[3]
            else:
                x_small_4 = 0

            x[B//2:] = x_small_1 + x_small_2 + x_small_3 + x_small_4
        
        return x, latency

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, alpha=[1, 2, 4, 8], stage = 0, depth = 0, num_heads = 4):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm_large = norm_layer(4 * dim)
        self.norm_small = norm_layer(4 * dim)
        self.alpha = alpha
        self.stage = stage
        self.depth = depth
        self.num_heads = num_heads

    def forward(self, x, gumbel_weights, latency = 0):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x_large, x_small = x[:B//2], x[B//2:]
        x_large = x_large.view(B//2, H, W, C)

        x_large0 = x_large[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x_large1 = x_large[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x_large2 = x_large[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x_large3 = x_large[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x_large = torch.cat([x_large0, x_large1, x_large2, x_large3], -1)  # B H/2 W/2 4*C
        x_large = x_large.view(B//2, -1, 4 * C)  # B H/2*W/2 4*C

        x_large = self.norm_large(x_large)
        x_large = self.reduction(x_large)

        x_small = x_small.view(B//2, H, W, C)

        x_small0 = x_small[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x_small1 = x_small[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x_small2 = x_small[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x_small3 = x_small[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x_small = torch.cat([x_small0, x_small1, x_small2, x_small3], -1)  # B H/2 W/2 4*C
        x_small = x_small.view(B//2, -1, 4 * C)  # B H/2*W/2 4*C

        x_small = self.norm_small(x_small)
        l = x_small.shape[2]

        if gumbel_weights[0] != 0 and self.alpha[0] <= self.num_heads:
            x_small_1 = torch.clone(x_small)
            x_small_1[:, :, l//self.alpha[0]:] = 0
            flops = H * W * (self.dim // self.alpha[0])
            latency += flops * gumbel_weights[0]
        else:
            x_small_1 = 0

        if gumbel_weights[1] != 0 and self.alpha[1] <= self.num_heads:
            x_small_2 = torch.clone(x_small)
            x_small_2[:, :, l//self.alpha[1]:] = 0
            flops = H * W * (self.dim // self.alpha[1])
            latency += flops * gumbel_weights[1]
        else:
            x_small_2 = 0

        if gumbel_weights[2] != 0 and self.alpha[2] <= self.num_heads:
            x_small_3 = torch.clone(x_small)
            x_small_3[:, :, l//self.alpha[2]:] = 0
            flops = H * W * (self.dim // self.alpha[2])
            latency += flops * gumbel_weights[2]
        else:
            x_small_3 = 0

        if gumbel_weights[3] != 0 and self.alpha[3] <= self.num_heads:
            x_small_4 = torch.clone(x_small)
            x_small_4[:, :, l//self.alpha[3]:] = 0
            flops = H * W * (self.dim // self.alpha[3])
            latency += flops * gumbel_weights[3]
        else:
            x_small_4 = 0
        
        x_small = x_small_1 + x_small_2 + x_small_3 + x_small_4

        x_small = self.reduction(x_small)
        l = x_small.shape[2]

        if gumbel_weights[0] != 0 and self.alpha[0] <= self.num_heads:
            x_small_1 = torch.clone(x_small)
            x_small_1[:, :, l//self.alpha[0]:] = 0
            flops = (H // 2) * (W // 2) * 4 * (self.dim // self.alpha[0]) * 2 * (self.dim // self.alpha[0])
            latency += flops * gumbel_weights[0]
        else:
            x_small_1 = 0

        if gumbel_weights[1] != 0 and self.alpha[1] <= self.num_heads:
            x_small_2 = torch.clone(x_small)
            x_small_2[:, :, l//self.alpha[1]:] = 0
            flops = (H // 2) * (W // 2) * 4 * (self.dim // self.alpha[1]) * 2 * (self.dim // self.alpha[1])
            latency += flops * gumbel_weights[1]
        else:
            x_small_2 = 0

        if gumbel_weights[2] != 0 and self.alpha[2] <= self.num_heads:
            x_small_3 = torch.clone(x_small)
            x_small_3[:, :, l//self.alpha[2]:] = 0
            flops = (H // 2) * (W // 2) * 4 * (self.dim // self.alpha[2]) * 2 * (self.dim // self.alpha[2])
            latency += flops * gumbel_weights[2]
        else:
            x_small_3 = 0

        if gumbel_weights[3] != 0 and self.alpha[3] <= self.num_heads:
            x_small_4 = torch.clone(x_small)
            x_small_4[:, :, l//self.alpha[3]:] = 0
            flops = (H // 2) * (W // 2) * 4 * (self.dim // self.alpha[3]) * 2 * (self.dim // self.alpha[3])
            latency += flops * gumbel_weights[3]
        else:
            x_small_4 = 0
        
        x_small = x_small_1 + x_small_2 + x_small_3 + x_small_4

        x = torch.cat((x_large, x_small), dim = 0)
        return x, latency

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, alpha = [1, 2, 4, 8], stage=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process,
                                 alpha=alpha,
                                 stage=stage,
                                 depth=i)
            for i in range(depth)])

        self.gumbel_weight = nn.Parameter(torch.rand(4))
        self.gumbel_noise = nn.Parameter(sample_gumbel(self.gumbel_weight.size()), requires_grad = False)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, alpha=alpha, num_heads=num_heads)
        else:
            self.downsample = None

    def forward(self, x, epoch, latency=0):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, latency = checkpoint.checkpoint(blk, x, epoch, latency=latency)
            else:
                x, latency = blk(x, epoch, latency=latency)
        
        gumbel_weights = gumbel_softmax(self.gumbel_weight, self.gumbel_noise, 15*((0.956)**epoch), False)

        if self.downsample is not None:
            x, latency = self.downsample(x, gumbel_weights, latency=latency)
        return x, latency

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x, latency=0):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        
        Ho, Wo = self.patches_resolution
        latency += Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            latency += Ho * Wo * self.embed_dim
        return x, latency

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, alpha=[1, 2, 4, 8], **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               fused_window_process=fused_window_process,
                               alpha=alpha,
                               stage=i_layer)
            self.layers.append(layer)

        self.norm_large = norm_layer(self.num_features)
        self.norm_small = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.alpha = alpha

        self.gumbel_weight = nn.Parameter(torch.rand(4))
        self.gumbel_noise = nn.Parameter(sample_gumbel(self.gumbel_weight.size()), requires_grad = False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, epoch, latency=0):
        x, latency = self.patch_embed(x, latency=latency)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x = torch.cat((x, x), dim = 0)

        gumbel_weights = gumbel_softmax(self.gumbel_weight, self.gumbel_noise, 15*((0.956)**epoch), False)

        for layer in self.layers:
            x, latency = layer(x, epoch, latency=latency)

        B, L, C = x.shape
        x_large = self.norm_large(x[:B//2])  # B L C
        x_large = self.avgpool(x_large.transpose(1, 2))  # B C 1
        x_large = torch.flatten(x_large, 1)

        x_small = self.norm_small(x[B//2:])
        if gumbel_weights[0] != 0:
            x_small_1 = torch.clone(x_small)
            x_small_1[:, :, C//self.alpha[0]:] = 0
            x_small_1 *= gumbel_weights[0]
            flops = self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // ((2 ** self.num_layers) * self.alpha[0])
            latency += flops * gumbel_weights[0]
        else:
            x_small_1 = 0

        if gumbel_weights[1] != 0:
            x_small_2 = torch.clone(x_small)
            x_small_2[:, :, C//self.alpha[1]:] = 0
            x_small_2 *= gumbel_weights[1]
            flops = self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // ((2 ** self.num_layers) * self.alpha[1])
            latency += flops * gumbel_weights[1]
        else:
            x_small_2 = 0

        if gumbel_weights[2] != 0:
            x_small_3 = torch.clone(x_small)
            x_small_3[:, :, C//self.alpha[2]:] = 0
            x_small_3 *= gumbel_weights[2]
            flops = self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // ((2 ** self.num_layers) * self.alpha[2])
            latency += flops * gumbel_weights[2]
        else:
            x_small_3 = 0

        if gumbel_weights[3] != 0:
            x_small_4 = torch.clone(x_small)
            x_small_4[:, :, C//self.alpha[3]:] = 0
            x_small_4 *= gumbel_weights[3]
            flops = self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // ((2 ** self.num_layers) * self.alpha[3])
            latency += flops * gumbel_weights[3]
        else:
            x_small_4 = 0

        x_small = x_small_1 + x_small_2 + x_small_3 + x_small_4

        x_small = self.avgpool(x_small.transpose(1, 2))  # B C 1
        x_small = torch.flatten(x_small, 1)

        x = torch.cat((x_large, x_small), dim = 0)
        return x, latency

    def forward(self, x, epoch, latency=0):
        x, latency = self.forward_features(x, epoch, latency=latency)
        B = x.shape[0]
        x_large = self.head(x[:B//2])
        x_small = self.head(x[B//2:])

        x = torch.cat((x_large, x_small), dim = 0)
        latency += self.num_features * self.num_classes
        return x, latency

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
