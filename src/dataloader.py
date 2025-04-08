from torch.utils.data import Dataset
from torchvision import transforms
from util.util import args
from util import util
import torch
import glob
import random


class StegDataset(Dataset):
    def __init__(self, img_dir, qr_dir):
        self.img_size = (args.train.img_size, args.train.img_size)
        self.img_dir = img_dir
        self.qr_dir = qr_dir
        self.img_name_list = glob.glob(self.img_dir)
        self.qr_name_list = glob.glob(self.qr_dir)
        
        self.img_transform = transforms.Compose([
            transforms.Resize(int(self.img_size[0] * 1.5)), 
            transforms.RandomCrop(self.img_size)
        ])
        self.qr_transform = transforms.Resize(self.img_size)
            
    
    def __len__(self):
        return len(self.img_name_list)
    
    
    def __repr__(self):
        return f'img num: {len(self.img_name_list)}\n' \
               f'qr num: {len(self.qr_name_list)}'
    
    
    def __getitem__(self, idx):
        img = util.image_to_tensor(self.img_name_list[idx])[0]
        qr_idx = random.randint(0, len(self.qr_name_list) - 1)
        qr = util.image_to_tensor(self.qr_name_list[qr_idx])[0, :1, ...]
        
        if (img.shape[0] == 1):
            img = torch.cat([img, img, img], dim=0)
        
        img = self.img_transform(img)
        qr = self.qr_transform(qr)[:1, ...]
        
        return {'img': img, 'qr': qr}
            
        
        
