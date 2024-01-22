from utils.data_RGB import get_validation_data
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as PSNR
import numpy as np
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils
from logger import *
import yaml
import scipy.io as sio
from tqdm import tqdm

with open('test.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

model_restoration = utils.get_arch(opt['MODEL'])

dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')


path_chk_rest = os.path.join(model_dir, opt['VAL']['PRETRAIN_MODEL'])
utils.load_checkpoint(model_restoration, path_chk_rest)
val_epoch = utils.load_start_epoch(path_chk_rest)

print("===>Testing using weights of epoch: ",val_epoch)
model_restoration.cuda()
#model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

psnr_val_rgb = []
file_path = os.path.join(opt['PATH']['VAL_DATASET'],'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(file_path)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /= 255
restored = np.zeros_like(Inoisy)
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i,k,:,:,:]).unsqueeze(0).permute(0,3,1,2).cuda()
            _,_,h,w = noisy_patch.shape
            factor = 32
            H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            noisy_patch = F.pad(noisy_patch,(0,padw,0,padh),'reflect')

            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch,0,1).cpu().detach().permute(0,2,3,1).squeeze(0)
            restored[i,k,:,:,:] = restored_patch
            
            if opt['VAL']['SAVE_IMG']:
                save_img_path = os.path.join(result_dir, 'epoch_' + str(val_epoch))
                file_name = 'denoisy_%04d_%02d.png'%(i+1,k+1)
                utils.mkdir(save_img_path)
                utils.save_img(os.path.join(save_img_path, file_name), img_as_ubyte(restored_patch))
sio.savemat(os.path.join(save_img_path,'Idenoised.mat'), {"Idenoised":restored})
