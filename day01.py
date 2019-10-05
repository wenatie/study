#!/usr/bin/env python
# coding: utf-8

# In[1]:
# 解决直接导入cv2的时候出现找不到cv2模块的问题
from cv2 import cv2 as cv2
# import cv2


# In[2]:


cv2.__version__


# In[3]:


import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# # 0为灰度图 1为彩色三通道

# In[5]:

# img = cv2.imread('D:/my_code_text/lena.jpg',0)
img = cv2.imread('D:/my_code_text/lena.jpg',1)


# In[6]:

cv2.imshow('lena',img)
# 设置waitKey避免bug
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[7]:


img_gray = cv2.imread('D:/my_code_text/lena.jpg',0)
cv2.imshow('lena',img_gray)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# # opencv的三通道是bgr matplotlib的三通道是rgb

# In[8]:

# 这是matplotlib的展示图片
plt.imshow(img)


# In[9]:


B,G,R = cv2.split(img)
img_new = cv2.merge((R,G,B))
plt.imshow(img_new)


# In[10]:


B,G,R = cv2.split(img)
const = 200
R[R>55] = 255
R[R<=55] = R[R<=55]+200
img_new = cv2.merge((R,G,B))
plt.imshow(img_new)


# In[11]:


img_dark = cv2.imread('D:/my_code_text/dark.jpg',1)
cv2.imshow('dark',img_dark)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[12]:


# gamma校正 使图片变的更亮或者更暗


# In[13]:

# 伽马校正函数 定义列表变量并设置列表元素和图片通道一一对应，利用numpy把列表转换成数组，利用LUT函数
'''LUT函数 void LUT(InputArray src, InputArray lut, OutputArray dst，int   interpolation);
    src表示的是输入图像(可以是单通道也可是3通道)
    lut表示查找表（查找表也可以是单通道，也可以是3通道，如果输入图像为单通道，那查找表必须为单通道,
        若输入图像为3通道，查找表可以为单通道，也可以为3通道，若为单通道则表示对图像3个通道都应用这个表，
        若为3通道则分别应用
    dst表示输出图像，
    interpolation表示插值，这个参数没有用，源代码里面也没有用它，默认为0就行，传其他值会报错。
    opencv3里面的LUT函数    void LUT(InputArray src, InputArray lut, OutputArray dst）;与二相比没有了最后一个参数
'''
def gm(img,gamma = 1.0):
    table = []
    for i in range(256):
        table.append(((i/255)**gamma)*255)
    table = np.array(table).astype('uint8')
    img_dark_new = cv2.LUT(img,table)
    return img_dark_new
image_new_dark = gm(img_dark,2)
cv2.imshow('dark',image_new_dark)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[14]:


# image crop 图片的切片


# In[15]:

# 利用shape函数获取图片的大小，然后直接切片
img_dark = cv2.imread('D:/my_code_text/dark.jpg',1)
print(img_dark.shape)
img_dark_crop = img_dark[0:50,0:60]
cv2.imshow('dark',img_dark_crop)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[16]:


img = cv2.imread('D:/my_code_text/lena.jpg',1)
img.shape


# In[17]:
'''
1、用于array（数组）对象
    a.flatten() 默认按行的方向降维
    a.flatten('F') #按列降维
    a.flatten('A') #按行降维
2、用于mat（矩阵）对象
    矩阵都会生成一个一行的或一列的二维数组
'''

img_hist = img.flatten()
img_hist.shape


# In[18]:


plt.hist(img_hist,256,[0,256])


# In[19]:
'''函数原型void equalizeHist(InputArray src,OutputArray dst)
    函数的作用：直方图均衡化，用于提高图像的质量；
'''

img_eq = cv2.equalizeHist(img_gray)
cv2.imshow('img_gray',img_eq)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[20]:


img_hist_new = img_eq.flatten()
plt.hist(img_hist_new,256,[0,256])


# In[21]:


# 图片的相似变换（包括平移、旋转和缩放）


# In[22]:
'''getRotationMatrix2D(center,angle,scale)
        center: 旋转中心点，一般取图片的中心，即宽的一半（shape[1]/2），高的一半（shape[0]/2）
        angle： 旋转的角度，正值为逆时针，负值为顺时针
        scale：缩放因子，即缩放比例
    warpAffine(img, M, (img.shape[1], img.shape[0]))
        img: 输入的图片
        M：变换矩阵，
        (img.shape): 变化后图片的大小
'''
M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),60,0.6)
img_rotate = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
cv2.imshow('img_rotate',img_rotate)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[23]:


# 图片的仿射变换（由平行四边形到平行四边形的变化）


# In[48]:
'''getAffineTransform(InputArray src, InputArray dst)
    src: 输入图像的三角形顶点坐标。
    dst: 输出图像的三角形顶点坐标。
    均需要三组点坐标
warpAffine(img,M,(cols,rows))
'''

img = cv2.imread('D:/my_code_text/lena.jpg',1)
rows, cols = img.shape[:2]
pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1]])
pts2 = np.float32([[cols*0.3,rows*0.1],[cols*0.9,rows*0.2],[cols*0.1,rows*0.9]])

M = cv2.getAffineTransform(pts1,pts2)
dst = cv2.warpAffine(img,M,(cols,rows))
cv2.imshow('dst',dst)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


# In[ ]:


# 图片的透视变换


# In[ ]:
'''getPerspectiveTransform(src,dst)
    src：输入四边形顶点坐标
    dst：输出四边形顶点坐标
warpPerspective(img,M,(cols,rows))
    img: 输入图片
    M: 透视变换矩阵
'''

img = cv2.imread('D:/my_code_text/lena.jpg',1)
cols, rows = img.shape[:2]
pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1],[rows-1,cols -1]])
pts2 = np.float32([[cols*0.3,rows*0.1],[cols*0.9,rows*0.2],[cols*0.1,rows*0.9],[rows-1,cols -1]])

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(cols,rows))
cv2.imshow('dst',dst)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

