import re
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
from natsort import natsorted
from glob import glob
import yaml
from tqdm import tqdm
import lpips

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
alex = lpips.LPIPS(net='alex').cuda()

filesL = natsorted(glob(os.path.join('/home/yoga/save_pth/DPDD/test', 'inputL', '*.png')))
filesR = natsorted(glob(os.path.join('/home/yoga/save_pth/DPDD/test', 'inputR', '*.png')))
filesC = natsorted(glob(os.path.join('/home/yoga/save_pth/DPDD/test', 'target', '*.png')))

indoor_labels  = np.load('/home/yoga/save_pth/DPDD/test/indoor_labels.npy')
outdoor_labels = np.load('/home/yoga/save_pth/DPDD/test/outdoor_labels.npy')
i = 1
psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileL, fileR, fileC in tqdm(zip(filesL, filesR, filesC), total=len(filesC)):

        imgL = np.float32(utils.load_img16(fileL))/65535.
        imgR = np.float32(utils.load_img16(fileR))/65535.
        imgC = np.float32(utils.load_img16(fileC))/65535.

        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0,3,1,2).cuda()
        patchL = torch.from_numpy(imgL).unsqueeze(0).permute(0,3,1,2)
        patchR = torch.from_numpy(imgR).unsqueeze(0).permute(0,3,1,2)

        input_ = torch.cat([patchL, patchR], 1).cuda()
        factor = 32
        h,w = input_.shape[2],input_.shape[3]
        H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_,(0,padw,0,padh),'reflect')
        restored = model_restoration(input_)
        restored = torch.clamp(restored,0,1)
        restored = restored[:,:,:h,:w]
        pips.append(alex(patchC, restored, normalize=True).item())

        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored_img = np.uint16((restored*65535).round())
        psnr.append(utils.PSNR(imgC, restored))
        mae.append(utils.MAE(imgC, restored))
        ssim.append(utils.SSIM(imgC, restored))
        if opt['VAL']['SAVE_IMG']:
            save_img_path = os.path.join(result_dir, 'epoch_' + str(val_epoch))
            file_name = str(i) + '_restored.png'
            i += 1
            utils.mkdir(save_img_path)
            utils.save_img(os.path.join(save_img_path, file_name), restored_img)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels-1], mae[indoor_labels-1], ssim[indoor_labels-1], pips[indoor_labels-1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels-1], mae[outdoor_labels-1], ssim[outdoor_labels-1], pips[outdoor_labels-1]

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae), np.mean(pips)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor), np.mean(mae_indoor), np.mean(pips_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor), np.mean(mae_outdoor), np.mean(pips_outdoor)))
