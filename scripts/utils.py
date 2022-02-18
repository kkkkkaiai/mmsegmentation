import numpy as np
import copy

# color ref https://www.sioe.cn/yingyong/yanse-rgb-16/
# classify ref https://blog.csdn.net/qq_36735489/article/details/96436763
# 1  building RoyalBlue  皇家蓝 	 #4169E1 	65,105,225
# 4  tree  SpringGreen 春天的绿色	 #3CB371 60,179,113
# 6  road * LightGrey	浅灰色	#D3D3D3	211,211,211
# 9  grass LawnGreen	草坪绿	#7CFC00	124,252,0
# 11 sidewalk pavement * 浅灰色	#D3D3D3	211,211,211
# 12 person individual someone somebody mortal soul DarkOrange 	深橙色 	#FF8C00 255,140,0
# 13 earth ground * 浅灰色	#D3D3D3	211,211,211
# 20 car auto automobile machine OrangeRed	橙红色	#FF4500	255,69,0
# #43 signboard sign
# 83 truck  OrangeRed	橙红色	#FF4500	255,69,0
classes = {1: [65., 105., 225.],
           4: [60., 179., 113.],
           6: [211., 211., 211.],
           9: [124., 252., 0.],
           11: [211., 211., 211.],
           12: [255, 140, 0],
           13: [211., 211., 211.],
           20: [255., 69., 0.],
           83: [255., 69., 0.]}
# 原始点云颜色显示
raw_color = np.array([123, 123, 123])/255


def revert_rgb(image):
    """
    转换bgr图像到rgb，也可以反想转换
    @in  rgb图像
    @out bgr图像
    """
    return image[..., ::-1]


def put_mask(img, mask, classes, alpha=0.5):
    """
    将分割的结果显示在图像上
    @in  原始图像，分割结果，欲显示的类别，透明度
    @out 带有分割结果的图像
    """
    temp_img = copy.copy(img)
    for _, key in enumerate(classes):
        row_list, col_list = np.where(mask == key)
        temp_img[row_list, col_list] = (img[row_list, col_list] * (1 - alpha) + np.array(classes[key]) * 0.3).astype(
            np.uint8)
    # temp_img = convert_rgb(temp_img)
    return temp_img

