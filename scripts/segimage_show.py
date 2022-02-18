# 显示分割后的效果

import os
from PIL import Image
import matplotlib.pyplot as plt

cur_dir = os.getcwd()
bin_dir = '/home/znfs/2021/DATASET/ROS/sync/image_bin'
file_name = '1631245044524575710'
bin_file = os.path.join(bin_dir, file_name)+'.jpg'

print(bin_file)
img = Image.open(bin_file).convert('P')

plt.imshow(img)
plt.show()
