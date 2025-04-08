import torch
import torch.nn as nn
from util.util import args
from util import util
import util.qr as uqr
from model.attnflow import AttnFlow
from torchvision import transforms
import os

device = 'cpu'
num_module = 37
img_size = 224

if __name__ == '__main__':
    os.makedirs('./result/', exist_ok=True)
    
    net = AttnFlow(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module)
    net.load_state_dict((torch.load('./pretrained/rmsteg_epoch_25.pth')))
    net = net.to(device)

    host_image = util.image_to_tensor('./test_img/test_host.png').to(device)
    host_image = transforms.Resize((img_size, img_size))(host_image)
    
    # generate and save qr code
    uqr.save_qr_code(
        save_dir='./result/qr.png',
        version=5,
        message='rmsteg'
    )
    qr = util.image_to_tensor('./result/qr.png').to(device)
    qr = transforms.Resize((img_size, img_size))(qr)[:, :1, ...]
    
    steg, trans_qr = net.encode(torch.cat([host_image, qr], dim=1))
    distort = net.distort(steg)
    decode_qr = net.decode(distort)
    
    # save results
    util.save_image_from_tensor(host_image, './result/test_host.png')
    util.save_image_from_tensor(qr, './result/test_qr.png')
    util.save_image_from_tensor(steg, './result/test_steg.png')
    util.save_image_from_tensor(torch.abs(steg - host_image), './result/steg_res.png')
    util.save_image_from_tensor(trans_qr, './result/test_trans_qr.png')
    util.save_image_from_tensor(distort, './result/test_distort.png')
    util.save_image_from_tensor(decode_qr, './result/test_decode_qr.png')
    util.save_image_from_tensor(util.get_error_map(qr, decode_qr, num_module=num_module), './result/test_qr_error.png')