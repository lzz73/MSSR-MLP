import os
from pickletools import UP_TO_NEWLINE
from pty import master_open
from turtle import up, update
from logger import ordered_yaml, dict2str, log
import yaml
with open('train.yml', mode='r') as f_yml:
    Loader, _ = ordered_yaml()
    opt = yaml.load(f_yml, Loader=Loader)

gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True
from visdom import Visdom
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math

from collections import OrderedDict
import random
import time
import datetime
from tqdm import tqdm
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as PSNR

import utils
from utils.data_RGB import get_training_data, get_validation_data
import utils.losses as losses

from ptflops import get_model_complexity_info


######### Logs dir ###########
dir_name = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(dir_name, 'log', opt['MODEL']['NAME'] + '_' + opt['MODEL']['MODE'])
utils.mkdir(log_dir)

train_log = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt['MODEL'])
num_params = 0
with open(train_log,'a') as f:
    f.write(dict2str(opt)+'\n')
    f.write(str(model_restoration)+'\n')
    
    for param in model_restoration.parameters():
        num_params += param.numel()
        
    f.write('parameters:' + str(num_params))
    

model_restoration.cuda()
macs_hand = model_restoration.flops()
log('macs_hand = %.2f GMac' % macs_hand, train_log,P=False)
macs, params = get_model_complexity_info(model_restoration, (3,256,256), as_strings=True,print_per_layer_stat = False, verbose=True)
log('macs = %s \t params = %s'%(macs, params),train_log)


######### Optimizer  ###########
start_epoch = 1
optimizer = torch.optim.Adam(model_restoration.parameters(), **opt['OPTIM'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **opt['SCHE'])
'''
def update_lr(epoch):
    current_epoch = epoch
    for param in optimizer.param_groups:
        lr = param['lr']
    lr_decay = ((1e-4)-(1e-7))/700
    lr_new = lr - lr_decay
    param['lr'] = lr_new
    print('----current_epoch : %d--------learning rate :%.10f---------------' % (epoch, lr_new))



if opt['TRAIN']['RESUME']:
        path_chk_rest = utils.get_last_path(model_dir, opt['TRAIN']['PRETRAIN_MODEL'])
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt['OPTIM']['lr']
        if start_epoch > 300:
            for i in range(1, start_epoch):
                update_lr(start_epoch)
    
        log('------------------------------------------------------------------------------',train_log)
        log("==> Resuming Training with learning rate:", train_log)
        log('------------------------------------------------------------------------------',train_log)
'''
######### Resume ###########
if opt['TRAIN']['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, opt['TRAIN']['PRETRAIN_MODEL'])
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for param in optimizer.param_groups:
        param['lr'] = opt['OPTIM']['lr']

    for i in range(1, 1436608):
        scheduler.step()
    new_lr = scheduler.get_last_lr()[0]
    
    log('------------------------------------------------------------------------------',train_log)
    log("==> Resuming Training with learning rate:%.10f"%new_lr, train_log)
    log('------------------------------------------------------------------------------',train_log)


######### Loss ###########
criterion_L1 = torch.nn.L1Loss()

######### DataLoaders ###########
train_dataset = get_training_data(opt['PATH']['TRAIN_DATASET'])
train_loader = DataLoader(dataset=train_dataset, batch_size=opt['TRAIN']['BATCH_SIZE'], shuffle=True, num_workers=16, drop_last=False, pin_memory=False)

val_dataset = get_validation_data(opt['PATH']['VAL_DATASET'], {'patch_size':opt['TRAIN']['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=False)

trainset_len = train_dataset.__len__()
valset_len = val_dataset.__len__()
total_iters = trainset_len/opt['TRAIN']['BATCH_SIZE']
log('trainset length: %d \ttotal iters per epoch: %d \tvalset length: %d'%(trainset_len, total_iters, valset_len), train_log)

######### Visdom ###########

log("------------------------------------------------------------------",train_log)
print('==> visdom initial')
window_loss = Visdom(port=opt['PORT']) 
window_blur = Visdom(port=opt['PORT'])
window_restored = Visdom(port=opt['PORT'])
window_sharp = Visdom(port=opt['PORT'])
window_loss.line([[0.,0.]], [0.], win='train_loss',opts=dict(title='train_loss',legend=['loss_L1','loss_FFT'],xlabel='epoch',ylabel='loss'))
log("------------------------------------------------------------------\n",train_log)

######### Train ###########
log("------------------------------------------------------------------",train_log)
log('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt['TRAIN']['TOTAL_EPOCHS']),train_log)
log("------------------------------------------------------------------",train_log)

best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, opt['TRAIN']['TOTAL_EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for iter, data in enumerate(train_loader, 1):

        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()
        restored = model_restoration(input_)
        # Compute loss 
        loss_L1 = criterion_L1(restored,target)
        
        restored_fft = torch.fft.rfft2(restored,dim=(-2,-1))
        target_fft = torch.fft.rfft2(target,dim=(-2,-1))
        #restored_fft = torch.rfft(restored,signal_ndim=2, normalized=False, onesided=False)
        #target_fft = torch.rfft(target,signal_ndim=2, normalized=False, onesided=False)
        loss_fft = criterion_L1(restored_fft,target_fft)
        
        loss = (loss_L1) + (loss_fft*0.1)
       
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_restoration.parameters(),0.01)
        optimizer.step()
        scheduler.step()
        epoch_loss +=loss.item()

        if iter % opt['TRAIN']['PRINT_FRE'] == 0:
            log("Epoch:%-4d \tIter:%-4d \tTime:%-4.f \tL1_loss:%.6f \tFFT_loss:%.6f \t LR:%.15f"%(epoch, iter, time.time()-epoch_start_time, loss_L1, loss_fft, scheduler.get_last_lr()[0]),train_log)
            
            step = epoch + iter/total_iters
            window_loss.line([[loss_L1.item(),loss_fft.item()]], [step], win = 'train_loss', update='append')
            window_blur.images(input_[0], nrow=1, win='Blur_Train', opts=dict(title='rainy'))
            window_restored.images(torch.clamp(restored[0],0,1), nrow=1, win='Restored_Train', opts=dict(title='Restored'))
            window_sharp.images(target[0], nrow=1, win='Sharp_Train', opts=dict(title='Sharp'))

        if iter % 1500 == 0:
            torch.save({'iter': iter, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_{str(epoch) + '_' + str(iter)}.pth")) 
            
    #### Save ####
    if epoch >=  opt['TRAIN']['SAVE']:
        if epoch % opt['TRAIN']['SAVE_FRE'] == 0:
            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 
    
    #### Evaluation ####
    if epoch >= opt['TRAIN']['VAL']:
        if epoch % opt['TRAIN']['VAL_FRE'] == 0:
            log('eval epoch %d psnr'%(epoch), train_log)
            model_restoration = torch.nn.DataParallel(model_restoration)
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(tqdm(val_loader), 1):
                target = img_as_ubyte(data_val[0].numpy().squeeze().transpose((1,2,0)))
                input_ = data_val[1].cuda()
                factor = 32
                h,w = input_.shape[2],input_.shape[3]
                H,W = ((h+factor)//factor)*factor,((w+factor)//factor)*factor
                padh = H-h if h%factor!=0 else 0
                padw = W-w if w%factor!=0 else 0
                input_ = F.pad(input_,(0,padw,0,padh),'reflect')
                ########padding to the multiple of 32
                with torch.no_grad():
                    restored = model_restoration(input_)

                restored = restored[0]
                restored = torch.clamp(restored,0,1)
                restored = restored[:,:h,:w]
                restored_img = img_as_ubyte(restored.cpu().numpy().squeeze().transpose((1,2,0)))
                log('%-6s \t %f' % (data_val[2][0], psnr_val_rgb[-1]), os.path.join(log_dir, 'val_' + str(epoch)+'.txt'), P=False)
                if opt['TRAIN']['SAVE_IMG']:
                    save_img_path = os.path.join(result_dir, 'epoch_' + str(epoch))
                    file_name = data_val[2][0] + '_restored.png'
                    utils.mkdir(save_img_path)
                    utils.save_img(os.path.join(save_img_path, file_name), restored_img)

    log("------------------------------------------------------------------",train_log)
    log("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f} LR: {:.10f}".format(epoch, time.time()-epoch_start_time, epoch_loss,scheduler.get_last_lr()[0]),train_log)
    log("------------------------------------------------------------------",train_log)
    

    
    '''
    if epoch > opt['TRAIN']['LR_RANGE']:
            update_lr(epoch)
    else:
        for param in optimizer.param_groups:
            lr = param['lr']
            print('current epoch: %d --------learning rate:%.10f--------' % (epoch,lr))
    '''
    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth")) 