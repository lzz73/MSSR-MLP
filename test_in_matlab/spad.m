clear all;
ts =0;
tp =0;
for i=1:1000                          % the number of testing samples
   x_true=im2double(imread(strcat('./test/SPAD/norain/',sprintf('norain-%d.png',i))));  % groundtruth 
   x_true = rgb2ycbcr(x_true);
   x_true = x_true(:,:,1); 
   x = im2double(imread(strcat('D:\Program Files\Polyspace\R2019b\bin\epoch_8_iter_298000\',sprintf('norain-%d_restored.png',i))));     %reconstructed image
   x = rgb2ycbcr(x);
   x = x(:,:,1);
   tp= tp+ psnr(x,x_true);
   ts= ts+ssim(x*255,x_true*255);
end
fprintf('psnr=%6.4f, ssim=%6.8f\n',tp/1000,ts/1000)
