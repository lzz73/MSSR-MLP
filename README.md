## Training
- You first need to prepare the datasets in the corresponding image restoration tasks.
- Different restoration tasks may require different training hyper-parameters. Please train the model with the arguments mentioned in the paper(An Efficient Multiscale Spatial Rearrangement MLP Architecture for Image Restoration) in different restoration tasks.

### Modify the way of data preprocessing
- You need to replace the name of 'from utils.dataset_RGB import ...' with 'from utils.dataset_RGB_xxxx import ...' in utils/data_RGB.py when you are conducting the training or testing of image defocus deblurring, image dehazing, image deraining and image gray-scale gaussian denoising tasks, as listed in utils.

### Before training the model, you need to open the visdom server in another command window to visualize the restoration results, please run
```
python -m visdom.server -port 4567
```

### Start training
- To train the model in image motion deblurring and image dehazing(indoor) tasks, or you want to train the model by epochs, please run 
```
python train_epoch.py
```
- To train the model in image defocus deblurring task, please run
```
python train_defocus.py
```
- To train the model in image gray-scale gaussian denoising task, please run 
```
python train_gaussian.py
```
- To train the model in image deraining and image dehazing(outdoor) tasks, or you want to train the model by iters, please run 
```
python train_iters.py
```
- To train the model in image denoising task, please run 
```
python train_sidd.py
```

## Evaluation

### The pretrained_weights will be released soon.

### Start evaluating
- You also need to prepare the testing dataset in the corresponding image restoration tasks
#### Testing on GoPro HIDE and SOTS(indoor, outdoor) datasets
- you just need to provide the weight and change the path of the dataset in the test.yml, and run
```
python test.py
```

#### Testing on DPDD dataset
- Following the same process above, and run
```
python test_defocus.py 
```

#### Testing on SIDD dataset
- Following the same process above, set the choice of 'SAVE_IMG' to True in test.yml to save the restoration results, and run
```
python test_sidd.py 
```
- To obtain PSNR/SSIM, run
```
test_in_matlab/sidd.m
```

#### Testing on DND dataset
- Following the same process above, set the choice of 'SAVE_IMG' to True to in test.yml to save the restoration results, and run
```
python test_dnd.py 
```
- To obtain PSNR/SSIM, send the restoration results to the DND website

#### Testing on SPAD dataset
- Following the same process above, set the choice of 'SAVE_IMG' to True to in test.yml to save the restoration results, and run
```
python test.py 
```
- To obtain PSNR/SSIM, run
```
test_in_matlab/spad.m
```

#### Testing on Set12 and BSD68(image gray-scale gaussian denoising task) datasets
- Following the same process above, set the choice of 'SAVE_IMG' to True in test.yml to save the restoration results, and run
```
python test_gaussian.py 
```
- To obtain PSNR, run 
```
python evaluate_gaussian.py
```

## Results 
### Restoration results in each restoration task will be released soon 
