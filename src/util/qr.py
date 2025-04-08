import qrcode
import string
import torch
import random
from torchvision import transforms
import util.util as util
import numpy as np
import scipy.stats as st


def get_gaussian_kernel(size, sigma=1.0):
    x = np.linspace(-sigma, sigma, size + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return torch.from_numpy(kern2d/kern2d.sum()).float().unsqueeze(0).unsqueeze(0)


def get_random_message(length):
    letters = string.ascii_lowercase + "0123456789:;'\\,.<>[]{}-=_+|?!@#$%^&*()"
    rand_string = ''.join(random.choice(letters) for i in range(length))
    return rand_string


def save_qr_code(save_dir, version=5, box_size=20, max_text=40, size=(256, 256), message=None):
    QRC = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=box_size,
        border=0
    )
    if message is None:
        message = get_random_message(random.randint(max_text - 10, max_text))
        
    QRC.add_data(message)
    QRC.make(fit=False)
    qr = QRC.make_image()

    qr = torch.from_numpy(np.array(qr)).unsqueeze(0).unsqueeze(0).float()
    qr = transforms.Resize(size)(qr)
    util.save_image_from_tensor(qr, save_dir)
    