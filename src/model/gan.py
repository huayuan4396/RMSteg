import torch
import torch.nn as nn
from model import common
from torchvision import transforms


class Discriminator(nn.Module):
    def __init__(self, data_dim=3):
        super(Discriminator, self).__init__()
        self.data_dim = data_dim

        self.net = nn.Sequential(
            nn.Conv2d(self.data_dim, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1, 4, stride=2, padding=1, bias=False),
        )
        
        
    @staticmethod
    def get_label_tensor(input, is_real):
        if is_real:
            tensor = torch.FloatTensor(1).fill_(1)
        else:
            tensor = torch.FloatTensor(1).fill_(0)
        tensor.requires_grad_(False)
        return tensor.expand_as(input).to(input.device)
        
        
    def calc_loss_dis(self, dis_result_real, dis_result_fake):
        label_real = self.get_label_tensor(dis_result_real, is_real=True)
        label_fake = self.get_label_tensor(dis_result_fake, is_real=False)
        
        loss_func = nn.L1Loss()
        dis_real_loss = loss_func(dis_result_real, label_real)
        dis_fake_loss = loss_func(dis_result_fake, label_fake)
        dis_loss = dis_real_loss + dis_fake_loss
        
        return dis_loss
    
    
    def calc_loss_net(self, dis_result):
        label_real = self.get_label_tensor(dis_result, is_real=True)
        
        loss_func = nn.L1Loss()
        dis_loss = loss_func(dis_result, label_real)
        
        return dis_loss


    def forward(self, x):
        x = transforms.Resize((256, 256))(x)
        dis_result = self.net(x)
        return dis_result