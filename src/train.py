import torch
import torch.nn as nn
from util.util import args
from util import util
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import StegDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import os
from model.gan import Discriminator
from util.qr import get_gaussian_kernel
from model.attnflow import AttnFlow
import lpips
from tqdm import tqdm

task_name = 'rmsteg'
num_module = 37
os.environ["CUDA_VISIBLE_DEVICES"] = args.train.cuda_devices
writer = SummaryWriter(log_dir=f'log/{task_name}/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    os.makedirs('./result/', exist_ok=True)
    os.makedirs('./log/', exist_ok=True)
    os.makedirs('./checkpoints/', exist_ok=True)
    
    # DDP preparation - model
    net = AttnFlow(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module).to(local_rank)
    # net.load_state_dict((torch.load('./pretrained/rmsteg.pth', map_location=f'cuda:{local_rank}'))) # if from checkpoint
    net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # DDP preparation - gan
    use_gan = args.train.use_gan
    if use_gan:
        dis = Discriminator().to(local_rank)
        dis = DDP(dis, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)
    
    calc_lpips = lpips.LPIPS(net='vgg').to(local_rank)
    
    # DDP preparation - dataset
    dataset = StegDataset(img_dir=args.data.train_img_dir, qr_dir=args.data.train_qr_dir)
    if dist.get_rank() == 0:
        print(dataset)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.train.batch_size, sampler=sampler)
    
    # optimizer - net
    params_trainable_net = list(filter(lambda p: p.requires_grad, list(net.parameters())))
    optimizer_net = optim.AdamW(params_trainable_net,
                                args.train.lr,
                                betas=(args.train.betas1, args.train.betas2),
                                weight_decay=args.train.weight_decay)
    lr_scheduler_net = optim.lr_scheduler.StepLR(optimizer_net, args.train.optim_step, gamma=args.train.optim_gamma)
    
    # optimizer - gan
    if use_gan:
        params_trainable_gan = (list(filter(lambda p: p.requires_grad, list(dis.parameters()))))
        optimizer_gan = optim.AdamW(params_trainable_gan,
                                    args.train.lr,
                                    betas=(args.train.betas1, args.train.betas2),
                                    weight_decay=args.train.weight_decay)
        lr_scheduler_gan = optim.lr_scheduler.StepLR(optimizer_net, args.train.optim_step, gamma=args.train.optim_gamma)
    
    iter_idx = 1
    for epoch_idx in range(1, args.train.epoch_num + 1):
        net.train()
        dataloader.sampler.set_epoch(epoch_idx)
        epoch_steg_loss = 0.0
        epoch_qr_loss = 0.0
        epoch_qr_fusion_loss = 0.0
        epoch_loss_dis = 0.0
        epoch_loss_dis_steg = 0.0
        epoch_lpips = 0.0
        epoch_ssim = 0.0
        epoch_attn_loss = 0.0
        epoch_s_time = time.time()
        
        for data_idx, data in tqdm(enumerate(dataloader)):
            
            img = data['img'].to(local_rank)
            qr = data['qr'].to(local_rank)
            
            # optimize discriminator
            if use_gan:
                steg, distort, decode_qr, fusion_qr = net(torch.cat([img, qr], dim=1))
                
                dis_result_real = dis(torch.cat([img], dim=1))
                dis_result_fake = dis(torch.cat([steg.detach()], dim=1))
                dis_loss = dis.module.calc_loss_dis(dis_result_real, dis_result_fake)
                epoch_loss_dis += dis_loss.item() * args.train.batch_size
                
                optimizer_gan.zero_grad()
                dis_loss.backward()
                optimizer_gan.step()
            
            # optimize net
            steg, distort, decode_qr, fusion_qr = net(torch.cat([img, qr], dim=1))
            steg_loss, ssim_loss, qr_loss, qr_fusion_loss, refine_qr = net.module.calc_loss(img, steg, qr, decode_qr, fusion_qr)
            lpips_loss = calc_lpips(img, steg).reshape(-1).mean()
        
            total_loss = steg_loss * 5.0 + ssim_loss * 0.1 + qr_loss * 20.0 + lpips_loss * 4.0
            
            if use_gan:
                dis_result = dis(torch.cat([steg], dim=1))
                dis_loss_steg = dis.module.calc_loss_net(dis_result)
                epoch_loss_dis_steg += dis_loss_steg.item() * args.train.batch_size

                total_loss += dis_loss_steg * 0.15
                
            optimizer_net.zero_grad()
            total_loss.backward()
            optimizer_net.step()
            
            epoch_steg_loss += steg_loss.item() * img.shape[0]
            epoch_qr_loss += qr_loss.item() * img.shape[0]
            epoch_lpips += lpips_loss.item() * img.shape[0]
            epoch_ssim += (1.0 - ssim_loss.item()) * img.shape[0]
            epoch_qr_fusion_loss += qr_fusion_loss.item() * img.shape[0]
            
            
            # save image
            if dist.get_rank() == 0 and data_idx % 20 == 0:
                if data_idx % 500 == 0:
                    torch.save(net.module.state_dict(), f'./checkpoints/{data_idx}.pth')
                    
                util.save_image_from_tensor(img, f'./result/epoch_{epoch_idx}_{data_idx}_cover.png')
                util.save_image_from_tensor(fusion_qr, f'./result/epoch_{epoch_idx}_{data_idx}_trans_qr.png')
                util.save_image_from_tensor(refine_qr, f'./result/epoch_{epoch_idx}_{data_idx}_refine_qr.png')
                util.save_image_from_tensor(steg, f'./result/epoch_{epoch_idx}_{data_idx}_steg.png')
                util.save_image_from_tensor(distort, f'./result/epoch_{epoch_idx}_{data_idx}_distort.png')
                util.save_image_from_tensor(qr, f'./result/epoch_{epoch_idx}_{data_idx}_qr.png')
                util.save_image_from_tensor(decode_qr , f'./result/epoch_{epoch_idx}_{data_idx}_decode_qr.png')
                util.save_image_from_tensor(util.get_error_map(qr, refine_qr, num_module=num_module) , f'./result/epoch_{epoch_idx}_{data_idx}_qr_error_map.png')
               
               
            # tensorboard logging
            writer.add_scalar('Loss/Steg Loss', epoch_steg_loss / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            writer.add_scalar('Loss/QR Loss', epoch_qr_loss / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            writer.add_scalar('Loss/QR Fusion Loss', epoch_qr_fusion_loss / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            if use_gan:
                writer.add_scalar('Loss/Dis Loss', epoch_loss_dis / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
                writer.add_scalar('Loss/Dis Steg Loss', epoch_loss_dis_steg / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
                
            writer.add_scalar('Metrices/SSIM', epoch_ssim / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            writer.add_scalar('Metrices/LPIPS', epoch_lpips / (data_idx * args.train.batch_size + img.shape[0]), iter_idx)
            
            iter_idx += 1
            
        lr_scheduler_net.step()
        if use_gan:
            lr_scheduler_gan.step()
        
        epoch_t_time = time.time()
        
        if dist.get_rank() == 0 and epoch_idx % args.train.save_freq == 0 or epoch_idx == 1:
            torch.save(net.module.state_dict(), f'./checkpoints/{task_name}_epoch_{epoch_idx}.pth')
            if use_gan:
                torch.save(dis.module.state_dict(), f'./checkpoints/{task_name}_dis_epoch_{epoch_idx}.pth')
        
        if dist.get_rank() == 0:
            print(f'epoch {epoch_idx} -- ' \
                f'lr: {optimizer_net.state_dict()["param_groups"][0]["lr"]:.8f}  ' \
                f'steg_loss: {epoch_steg_loss * torch.cuda.device_count() / len(dataset):.8f}  ' \
                f'ssim: {epoch_ssim * torch.cuda.device_count() / len(dataset):.8f}  ' \
                f'qr_loss: {epoch_qr_loss * torch.cuda.device_count() / len(dataset):.8f}  ' \
                f'lpips: {epoch_lpips * torch.cuda.device_count() / len(dataset):.8f}  ' \
                f'time: {epoch_t_time - epoch_s_time:.1f}')