import util.distortion as distortion
from util.util import args
import torch
import torch.nn.functional as F
import random
import torch.nn as nn
import numpy as np
import torchgeometry


def distortion_layer(img):
    rand_bright = random.random() * args.distortion.bright
    rand_hue = random.random() * args.distortion.hue
    rand_saturation = random.random() * args.distortion.saturation
    rand_jpeg = 100.0 - random.random() * (100 - args.distortion.jpeg_quality)
    rand_noise = random.random() * args.distortion.noise
    contrast_range = [args.distortion.contrast_l, args.distortion.contrast_h]
    
    if args.distortion.transition > 0:
        trans_mat = distortion.get_rand_transform_matrix(image_size=args.train.img_size, d=np.floor(args.train.img_size * np.random.uniform() * args.distortion.transition), batch_size=1).repeat(img.shape[0], 1, 1, 1).to(img.device)
        mask = torch.ones_like(img)
        trans_img = torchgeometry.warp_perspective(img, trans_mat[:, 1, :, :], dsize=(args.train.img_size, args.train.img_size), flags='bilinear')
        trans_mask = torchgeometry.warp_perspective(mask, trans_mat[:, 1, :, :], dsize=(args.train.img_size, args.train.img_size), flags='bilinear')
        img = (1.0 - trans_mask) + trans_img
    
    # blur
    blur_kernel = distortion.random_blur_kernel(probs=[0.25, 0.25],
                                                N_blur=args.distortion.blur_kernel_size,
                                                sigrange_gauss=[1.0, 3.0],
                                                sigrange_line=[0.25, 1.0],
                                                wmin_line=3).to(img.device)
    img = F.conv2d(img, blur_kernel, bias=None, padding=int((args.distortion.blur_kernel_size - 1) // 2))
    
    # noise
    noise = torch.normal(mean=0.0, std=rand_noise, size=img.shape, dtype=torch.float32).to(img.device)
    img = (img + noise).clamp(0.0, 1.0)
    
    # brightness & contrast
    rand_bright = distortion.get_rnd_brightness_torch(rand_bright, rand_hue, img.shape[0]).to(img.device)
    contrast_scale = torch.Tensor(img.shape[0]).uniform_(contrast_range[0], contrast_range[1])
    contrast_scale = contrast_scale.reshape(img.shape[0], 1, 1, 1).to(img.device)
    img = (img * contrast_scale + rand_bright).clamp(0.0, 1.0)
    
    # saturation
    saturation_scale = torch.FloatTensor([0.3, 0.6, 0.1]).reshape(1, 3, 1, 1).to(img.device)
    img_lum = torch.mean(img * saturation_scale, dim=1).unsqueeze_(1)
    img = (1.0 - rand_saturation) * img + rand_saturation * img_lum
    
    # jpeg
    if args.distortion.use_jpeg:
        img = distortion.jpeg_compress_decompress(img.cpu(),
                                                  rounding=distortion.round_only_at_0,
                                                  quality=rand_jpeg).to(img.device)
        
    return img
    
    