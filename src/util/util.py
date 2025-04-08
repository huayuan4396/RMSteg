import torch
from torchvision import transforms, utils
from PIL import Image
from easydict import EasyDict
import yaml
import matplotlib.pyplot as plt
import torch.nn as nn
import util.qr as uqr
from einops import rearrange
from pyzbar import pyzbar


def resize_qr_unit(qr_unit, factor):
    unit_size = qr_unit.shape[-1]
    qr_unit = qr_unit.reshape(qr_unit.shape[0], qr_unit.shape[1], -1, 1)
    qr_unit = qr_unit.repeat(1, 1, 1, factor * factor)
    qr_unit = rearrange(qr_unit, 'b c (h w) (f1 f2) -> b c (h f1) (w f2)', h=unit_size, w=unit_size, f1=factor, f2=factor)
    return qr_unit


def save_image_from_tensor(t, filename):
    while len(t.shape) < 4:
        t = t.unsqueeze(0)
    if t.shape[-1] == 1 or t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2)
    t = t.clone().detach()
    t = t.to(torch.device('cpu'))
    utils.save_image(t, filename)
    
    
def image_to_tensor(img_path):
    rgb = transforms.ToTensor()(Image.open(img_path))
    rgb = rgb.unsqueeze(0)
    if rgb.shape[1] == 4:
        rgb = rgb[:, :3, ...]
    return rgb
        
    
def calc_psnr(tensor1, tensor2):
    l1_loss = nn.L1Loss()
    l1 = l1_loss(tensor1, tensor2)
    psnr = 20 * torch.log10(1 / l1)
    return psnr
    
    
def calc_emr(qr, decode_qr, num_module):
    qr_trans = transforms.Resize((num_module * 5, num_module * 5))
    qr = qr_trans(qr)[:, :1, ...]
    decode_qr = qr_trans(decode_qr)[:, :1, ...]
    gs_kernel = uqr.get_gaussian_kernel(5, 1.0)
    gs_conv = nn.Conv2d(1, 1, 5, padding=0, stride=5).requires_grad_(False)
    gs_conv.weight.data = gs_kernel
    gs_conv = gs_conv.to(qr.device)
    qr = gs_conv(qr.clamp(0.0, 1.0))
    decode_qr = gs_conv(decode_qr.clamp(0.0, 1.0))
    error_map = torch.round(torch.abs(qr - decode_qr))
    emr = error_map.mean().reshape(-1).item()
    return emr * 100 # in percentage


def calc_tra(decode_qr):
    decode_qr = torch.cat([decode_qr, decode_qr, decode_qr], dim=1).clamp(0.0, 1.0)[0].cpu()[:3, ...]
    decode_qr = transforms.ToPILImage()(decode_qr)
    try:
        decode_qr_msg = pyzbar.decode(decode_qr, symbols=[pyzbar.ZBarSymbol.QRCODE])[0].data.decode("utf-8")
        return 1
    except Exception as e:
        return 0


    
def get_error_map(gt, pred, num_module):
    red = transforms.Resize((args.train.img_size, args.train.img_size))(image_to_tensor('./test_img/misc/red.png')).to(pred.device)
    green = transforms.Resize((args.train.img_size, args.train.img_size))(image_to_tensor('./test_img/misc/green.png')).to(pred.device)
    error = torch.abs(gt - pred)
    error = transforms.Resize((num_module, num_module))(error)
    error = resize_qr_unit(error, factor=7)
    error = transforms.Resize((args.train.img_size, args.train.img_size))(error)
    error_map = torch.round(error) * red + (1.0 - torch.round(error)) * green
    return error_map


with open('config.yaml', 'r') as f:
    args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    
    
if __name__ == '__main__':
    print(args)