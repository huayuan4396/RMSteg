import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models
from torchvision import transforms
from util.util import args
import numpy as np
from util import util
    
        
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
        
        
class Attention(nn.Module):
    def __init__(self, dim, num_head=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        self.hidden_dim = dim_head * num_head
        self.num_head = num_head
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, self.hidden_dim * 3, bias=False)
        self.proj_out = nn.Sequential(
            nn.Linear(self.hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x):
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_head), qkv)
        
        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(qk))
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj_out(out)
        
        return out
    
    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_head=8, dim_head=64, dropout=0.0):
        super(CrossAttention, self).__init__()
        self.hidden_dim = dim_head * num_head
        self.num_head = num_head
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_q = nn.Linear(dim, self.hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim, self.hidden_dim * 2, bias=False)
        self.proj_out = nn.Sequential(
            nn.Linear(self.hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.cnt = 0

    def forward(self, q, kv):
        q = self.norm(q)
        kv = self.norm(kv)

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_head), (q, k, v))
        
        qk = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.softmax(qk))
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj_out(out)
        
        return out

        
        
class Transformer(nn.Module):
    def __init__(self, dim, depth, num_head, dim_head, dim_mlp, dropout=0.0):
        super(Transformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer_list = nn.ModuleList()
        
        for _ in range(depth):
            self.layer_list.append(nn.ModuleList([
                Attention(dim, num_head, dim_head, dropout),
                FeedForward(dim, dim_mlp, dropout)
            ]))
            
    
    def forward(self, x):
        for attention, feedforward in self.layer_list:
            x = attention(x) + x
            x = feedforward(x) + x
            
        return self.norm(x)
    

class CATransformer(nn.Module):
    def __init__(self, dim, depth, num_head, dim_head, dim_mlp, dropout=0.0):
        super(CATransformer, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.layer_list = nn.ModuleList()
        
        for _ in range(depth):
            self.layer_list.append(nn.ModuleList([
                CrossAttention(dim, num_head, dim_head, dropout),
                FeedForward(dim, dim_mlp, dropout)
            ]))
            
    def forward(self, q, kv):
        for cross_attention, feedforward in self.layer_list:
            q = cross_attention(q, kv) + q
            q = feedforward(q) + q
            
        return self.norm(q)

    
    
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, dim, depth, num_head, dim_mlp, num_channel=3, dim_head=64, dropout=0.0, emb_dropout=0.0, use_cls_token=False):
        super(ViT, self).__init__()
        
        img_height, img_width = img_size
        patch_height, patch_width = patch_size
        assert img_height % patch_height == 0 and img_width % patch_width == 0, f'{img_height}, {patch_height}, {img_width}, {patch_width}'
        
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.img_height = img_height
        self.img_width = img_width
        
        num_patch = (img_height // patch_height) * (img_width // patch_width)
        dim_patch = num_channel * patch_height * patch_width
        
        self.dim = dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(dim_patch),
            nn.Linear(dim_patch, self.dim),
            nn.LayerNorm(self.dim)
        )
        
        self.cls_token_channel = 4
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + self.cls_token_channel if use_cls_token else num_patch, self.dim))
        
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, self.cls_token_channel, dim))
        
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, num_head, dim_head, dim_mlp, dropout)
        

    def patch_to_image(self, x):
        # x: (b, num_patch, c * patch_height * patch_width)
        # imgs: (b, c, h, w)
        
        x = x.reshape(x.shape[0], self.img_height // self.patch_height, self.img_width // self.patch_width, self.patch_height, self.patch_width, -1)
        x = torch.einsum('n h w p q c -> n c h p w q', x)
        img = x.reshape(shape=(x.shape[0], -1, self.img_height, self.img_width))
        return img


    def forward(self, x):
        x = self.to_patch_embedding(x)
        
        if self.use_cls_token:
            cls_token = repeat(self.cls_token, '1 n d -> b n d', b = x.shape[0])
            x = torch.cat([cls_token, x], dim=1)
    
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        return x


class InvertibleTokenShuffle(nn.Module):
    def __init__(self, num_channels, img_size=144, patch_size=8):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, is_rev):
        w_shape = self.w_shape
        if not is_rev:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float().view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, x, is_rev):
        weight = self.get_weight(is_rev)
        x = x.unsqueeze(-1)
        z = F.conv2d(x, weight)
        z = z.squeeze(-1)
        return z


class TransFlowBlock(nn.Module):
    def __init__(self, dim=768, dim_mlp=2048, clamp=2.0, id=''):
        super().__init__()
        self.clamp = clamp
        self.id = id
        
        # ρ
        self.r = nn.Sequential(
            Transformer(dim, 1, num_head=8, dim_head=64, dim_mlp=dim_mlp, dropout=0.0)
        )
        # η
        self.y = nn.Sequential(
            Transformer(dim, 1, num_head=8, dim_head=64, dim_mlp=dim_mlp, dropout=0.0)
        )
        # φ
        self.f = nn.Sequential(
            Transformer(dim, 1, num_head=8, dim_head=64, dim_mlp=dim_mlp, dropout=0.0)
        )
        
        self.cat = CATransformer(dim, 1, num_head=8, dim_head=64, dim_mlp=dim_mlp, dropout=0.0)
        self.scale = torch.nn.Parameter(torch.tensor(0.01))

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, c, is_rev):
        x1 = x[:, :x.shape[1] // 2, ...]
        x2 = x[:, x.shape[1] // 2:, ...]

        if not is_rev:
            t2 = self.f(x2)
            y1 = x1 + t2 + self.cat(x2, c) * self.scale
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = x1 - t2 - self.cat(y2, c) * self.scale

        return torch.cat([y1, y2], dim=1)
        
        
class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """
    def __init__(self, param_dim=(1,3,1,1)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x, is_rev):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, False) + 1e-6).view_as(self.bias)
            self.initialized += 1
        
        if not is_rev:
            z = (x - self.bias) / self.scale
            return z
        else:
            return x * self.scale + self.bias
    
    
class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, gc=16, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class ConvFlowBlock(nn.Module):
    def __init__(self, split_1 = 3, split_2 = 1, clamp=2.0):
        super().__init__()
        self.clamp = clamp
        
        self.split_1 = split_1
        self.split_2 = split_2
        
        # ρ
        self.r = nn.Sequential(
            DenseBlock(split_1, split_2, gc=16)
        )
        # η
        self.y = nn.Sequential(
            DenseBlock(split_1, split_2, gc=16)
        )
        # φ
        self.f = nn.Sequential(
            DenseBlock(split_2, split_1, gc=16)
        )
        
    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, is_rev):
        x1 = x[:, :self.split_1, ...]
        x2 = x[:, self.split_1:, ...]

        if not is_rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)