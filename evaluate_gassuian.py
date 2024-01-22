## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
import argparse
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import utils
from logger import log
from logger import *
import math

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def proc(filename):
    tar,prd = filename
    tar_img = load_gray_img(tar)
    prd_img = load_gray_img(prd)
        
    PSNR = calculate_psnr(tar_img, prd_img)
    #SSIM = calculate_ssim(tar_img, prd_img)
    return PSNR

#parser = argparse.ArgumentParser(description='Gasussian Color Denoising using Restormer')

#parser.add_argument('--model_type', required=True, choices=['non_blind','blind'], type=str, help='blind: single model to handle various noise levels. non_blind: separate model for each noise level.')
#parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')

#args = parser.parse_args()
with open('test.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
sigma_test = 15
val_iter = 1000
datasets = ['Set12', 'BSD68', 'Urban100']

for dataset in datasets:

    gt_path = os.path.join(r'/home/hua/Code/DFWB/test/', dataset)
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.tif')))
    assert len(gt_list) != 0, "Target files not found"

    
    file_path = os.path.join(r'./log/UNet_mlp/results/', dataset, str(sigma_test))
    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.tif')))
    assert len(path_list) != 0, "Predicted files not found"

    psnr, ssim = [], []
    img_files =[(i, j) for i,j in zip(gt_list,path_list)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            psnr.append(PSNR_SSIM)
            #ssim.append(PSNR_SSIM[1])

    avg_psnr = sum(psnr)/len(psnr)
    #avg_ssim = sum(ssim)/len(ssim)

    #print('For {:s} dataset Noise Level {:d} PSNR: {:f}\n'.format(dataset, sigma_test, avg_psnr))
    log("Dataset: {:s}\tEpoch: {}\tSigma_test: {}\tPSNR: {:.4f}".format(dataset, val_iter, sigma_test, avg_psnr),os.path.join(log_dir, 'val_' + str(val_iter)+'.txt'))
