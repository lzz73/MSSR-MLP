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
from natsort import natsorted
from glob import glob
import cv2

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)



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
val_iter = utils.load_start_iter(path_chk_rest)

sigma_test = 15
#sigmas = np.int_(sigmas_test.split(','))

factor = 32

print("===>Testing using weights of iter: ",val_iter)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


datasets = ['Set12', 'BSD68']
input_dir = opt['PATH']['VAL_DATASET']

print("Compute results for noise level",sigma_test)

for dataset in datasets:
    inp_dir = os.path.join(input_dir, dataset)
    #print(inp_dir)
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
    result_dir_tmp = os.path.join(result_dir, dataset, str(sigma_test))
    os.makedirs(result_dir_tmp, exist_ok=True)

    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img = np.float32(load_gray_img(file_))/255.

            np.random.seed(seed=0)  # for reproducibility
            img += np.random.normal(0, sigma_test/255., img.shape)

            img = torch.from_numpy(img).permute(2,0,1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
            save_gray_img(save_file, img_as_ubyte(restored))
