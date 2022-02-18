# 使用segformer来进行语义分割

from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import classes, put_mask, revert_rgb

model_path = '../models/segformer'
mmcv.mkdir_or_exist(model_path)

model_choose = 5

if model_choose == 3:
    download_link = 'https://download.openmmlab.com/mmsegmentation/v0.5/' \
                    'segformer/segformer_mit-b3_512x512_160k_ade20k/' \
                    'segformer_mit-b3_512x512_160k_ade20k_20210726_081410-962b98d2.pth'
    checkpoint_file = os.path.join(model_path, download_link.split('/')[-1])
    config_file = '../configs/segformer/segformer_mit-b3_512x512_160k_ade20k.py'

else:
    download_link = 'https://download.openmmlab.com/mmsegmentation/v0.5/' \
                    'segformer/segformer_mit-b5_640x640_160k_ade20k/' \
                    'segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
    checkpoint_file = os.path.join(model_path, download_link.split('/')[-1])
    config_file = '../configs/segformer/segformer_mit-b5_512x512_160k_ade20k.py'

try:
    mmcv.check_file_exist(checkpoint_file)
except:
    os.system('wget -P ' + model_path + ' ' + download_link)

model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

img_path = '/home/znfs/2021/DATASET/ROS/hk_image/1400.jpg'

img = mmcv.imread(img_path)
result = inference_segmentor(model, img)
row, col, channel = img.shape
result = np.asarray(result).reshape((row, col))

result_img = put_mask(img, result, classes, 0.5)
plt.imshow(result_img)
plt.show()
