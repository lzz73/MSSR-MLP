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
import time
class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count

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

val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['VAL']['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

psnr_val_rgb = []
adder = Adder()
for ii, data_val in enumerate(val_loader, 1):
    target_img = img_as_ubyte(data_val[0].numpy().squeeze().transpose((1,2,0)))
    input_ = data_val[1].cuda()
    factor = 32
    h,w = input_.shape[2],input_.shape[3]
    H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    input_ = F.pad(input_,(0,padw,0,padh),'reflect')

    with torch.no_grad():
        tm = time.time()
        restored = model_restoration(input_)
        elapse = time.time() - tm
        adder(elapse)

    restored = restored[0]
    restored = torch.clamp(restored,0,1)
    restored = restored[:,:h,:w]
    restored_img = img_as_ubyte(restored.cpu().numpy().squeeze().transpose((1,2,0)))

    ######### release testing gpu memory ##########
    del restored
    del input_
    torch.cuda.empty_cache()

    psnr_val_rgb.append(PSNR(restored_img, target_img))
    log('%-6s \t %f' % (data_val[2][0], psnr_val_rgb[-1]), os.path.join(log_dir, 'val_' + str(val_epoch)+'.txt'), P=True)
    if opt['VAL']['SAVE_IMG']:
        save_img_path = os.path.join(result_dir, 'epoch_' + str(val_epoch))
        file_name = data_val[2][0] + '_restored.png'
        utils.mkdir(save_img_path)
        utils.save_img(os.path.join(save_img_path, file_name), restored_img)

avg_psnr  = sum(psnr_val_rgb)/ii
log('total images = %d \t avg_psnr = %f' % (ii,avg_psnr), os.path.join(log_dir, 'val_' + str(val_epoch)+'.txt'), P=True)
print("Average time: %f" % adder.average())