import torch
import torch.nn as nn
from util.util import args
from model import common
import torchvision
from model.distortion_layer import distortion_layer
from einops.layers.torch import Rearrange
from util import util
from model.pytorch_ssim import SSIM
from einops import rearrange
from model import common
from torchvision import transforms
from util.qr import get_gaussian_kernel
from model.unet import UNet
import torch.nn.init as init
import random
import lpips


class AttnFlow(nn.Module):
    def __init__(self, block_num, use_itf=True, use_qr_trans=True, num_module=37):
        super(AttnFlow, self).__init__()
        
        # num_module indicates the module number in the QR Code
        # our full model use QR Code Version 5, which has 37 * 37 modules
        
        patch_size = 16
        dim = 768
        dim_mlp = 2048
        latent_dim = 64
        dim_patch = patch_size * patch_size
        img_size = args.train.img_size
        self.use_itf = use_itf
        self.use_qr_trans = use_qr_trans
        self.num_module = num_module
        
        if use_qr_trans:
            conv_flow_block_num = 2
            self.conv_flow_block_list = nn.ModuleList()
            for _ in range(conv_flow_block_num):
                self.conv_flow_block_list.append(common.ConvFlowBlock())
        
        if use_itf:
            self.qr_token_shuffle = common.InvertibleTokenShuffle(img_size // patch_size * img_size // patch_size)
        
        self.enhance_steg = UNet(3, 3)
        
        self.vit_img = nn.Sequential(
            nn.Conv2d(3, latent_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1),
            nn.GELU(),
            common.ViT(img_size=(img_size, img_size), patch_size=(patch_size, patch_size), dim=dim, depth=2, num_head=dim // 64, dim_mlp=dim_mlp, num_channel=latent_dim),
            nn.LayerNorm(dim),
        )
        self.vit_qr = nn.Sequential(
            common.ViT(img_size=(img_size, img_size), patch_size=(patch_size, patch_size), dim=dim, depth=2, num_head=dim // 64, dim_mlp=dim_mlp, num_channel=1),
            nn.LayerNorm(dim)
        )
        self.vit_steg = nn.Sequential(
            nn.Conv2d(3, latent_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1),
            nn.GELU(),
            common.ViT(img_size=(img_size, img_size), patch_size=(patch_size, patch_size), dim=dim, depth=2, num_head=dim // 64, dim_mlp=dim_mlp, num_channel=latent_dim),
            nn.LayerNorm(dim),
        )
        
        self.c_tokenizer = common.ViT(img_size=(img_size, img_size), patch_size=(patch_size, patch_size), dim=dim, depth=1, num_head=8, dim_mlp=dim_mlp, num_channel=3)
        
        self.trans_flow_block_list = nn.ModuleList()
        self.trans_act_norm_list = nn.ModuleList()
        
        for idx in range(block_num):
            self.trans_flow_block_list.append(common.TransFlowBlock(dim=dim, dim_mlp=dim_mlp, id=idx))
            self.trans_act_norm_list.append(common.Actnorm(param_dim=[1, img_size * img_size // patch_size // patch_size * 2, 1]))
            
        self.encode_proj_out = nn.Sequential(
            nn.Linear(dim, dim_patch * latent_dim),
            Rearrange('b (p1 p2) (c h w) -> b c (p1 h) (p2 w)', p1=img_size // patch_size, p2=img_size // patch_size, h=patch_size, w=patch_size),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(latent_dim, 3, 3, 1, 1)
        )
        
        self.decode_img_proj_out = nn.Sequential(
            nn.Linear(dim, dim_patch * latent_dim),
            Rearrange('b (p1 p2) (c h w) -> b c (p1 h) (p2 w)', p1=img_size // patch_size, p2=img_size // patch_size, h=patch_size, w=patch_size),
            nn.Conv2d(latent_dim, latent_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(latent_dim, 3, 3, 1, 1)
        )
        
        self.decode_qr_proj_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_patch * 1),
            Rearrange('b (p1 p2) (c h w) -> b c (p1 h) (p2 w)', p1=img_size // patch_size, p2=img_size // patch_size, h=patch_size, w=patch_size)
        )
        
        gs_kernel = get_gaussian_kernel(5, 1.0)
        self.qr_conv = nn.Conv2d(1, 1, 5, padding=0, stride=5).requires_grad_(False)
        self.qr_conv.weight.data = gs_kernel
        
        # distortion
        self.distort = distortion_layer
        
    
    def encode(self, x):
        img = x[:, :3, ...]
        bs = img.shape[0]
        
        if self.use_qr_trans:
            for conv_flow_block in self.conv_flow_block_list:
                x = conv_flow_block(x, is_rev=False)
        
        qr = x[:, 3:, ...]
        img_token = self.vit_img(img)
        
        c_token = self.c_tokenizer(img)
        qr_token = self.vit_qr(qr)
        if self.use_itf:
            qr_token = self.qr_token_shuffle(qr_token, is_rev=False)
        token = torch.cat([img_token, qr_token], dim=1)
        
        for trans_flow_block, trans_act_norm in zip(self.trans_flow_block_list, self.trans_act_norm_list):
            token = trans_act_norm(token, is_rev=False)
            token = trans_flow_block(token, c=c_token, is_rev=False)
            
        steg = self.encode_proj_out(token[:, :token.shape[1] // 2, ...])
            
        return steg.clamp(0.0, 1.0), qr
            
            
    def decode(self, x):
        img = x
        bs = img.shape[0]
        
        img_enhance = self.enhance_steg(img)
        
        img_token = self.vit_steg(img_enhance)
        c_token = self.c_tokenizer(img)
        qr_token = torch.randn_like(img_token)
        token = torch.cat([img_token, qr_token], dim=1)
        
        for trans_flow_block, trans_act_norm in zip(reversed(self.trans_flow_block_list), reversed(self.trans_act_norm_list)):
            token = trans_flow_block(token, c=c_token, is_rev=True)
            token = trans_act_norm(token, is_rev=False)
        
        qr_token = token[:, token.shape[1] // 2:, ...]
        if self.use_itf:
            qr_token = self.qr_token_shuffle(qr_token, is_rev=True)
            
        x = torch.cat([self.decode_img_proj_out(token[:, :token.shape[1] // 2, ...]), self.decode_qr_proj_out(qr_token)], dim=1)
        
        if self.use_qr_trans:
            for conv_flow_block in reversed(self.conv_flow_block_list):
                x = conv_flow_block(x, is_rev=True)
            
        return x[:, 3:, ...]
        
        
    def calc_loss(self, cover, steg, qr, decode_qr, fusion_qr):
        loss_func = nn.L1Loss()
        steg_loss = loss_func(cover, steg)
       
        qr_resize = self.qr_conv(transforms.Resize((5 * self.num_module, 5 * self.num_module))(qr))
        qr_fusion_resize = self.qr_conv(transforms.Resize((5 * self.num_module, 5 * self.num_module))(fusion_qr))
        qr_fusion_error = torch.abs(torch.round(qr_resize) - torch.round(qr_fusion_resize.clamp(0.0, 1.0) - 0.1))
        qr_fusion_loss = loss_func(qr_resize * qr_fusion_error, qr_fusion_resize * qr_fusion_error)
    
        qr_loss = loss_func(qr, decode_qr)
        
        ssim = SSIM()
        ssim_loss = 1.0 - ssim(steg, cover)
        
        refine_qr = util.resize_qr_unit(torch.round(self.qr_conv(transforms.Resize((5 * self.num_module, 5 * self.num_module))(decode_qr)).clamp(0.0, 1.0)), factor=5)
        refine_qr = transforms.Resize((args.train.img_size, args.train.img_size))(refine_qr)
        
        return steg_loss, ssim_loss, qr_loss, qr_fusion_loss, refine_qr
    
    
    def forward(self, x):
        steg, fusion_qr = self.encode(x)
        distort = self.distort(steg)
        decode_qr = self.decode(distort)
        
        return steg, distort, decode_qr, fusion_qr