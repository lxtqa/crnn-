import cv2
import numpy as np
import itertools
import os
from tqdm import tqdm
imname = "jfe"
impath = "./" + imname +".jpg"
originImg = cv2.imread(impath)

grayImg = cv2.cvtColor(originImg,cv2.COLOR_RGB2GRAY)


blurMethodList = [0,1,2]
blurKsizeArgList = [3]

thresholdMethodList = [0,1,2,3,4]
thresholdArgList = [63,127,191]#[31,63,95,127,159,191,223]
thresholdBlocksizeArgList = [11,31,51]
thresholdCArgList = [-10,1,10]

morphologyExMethodList = [0,1,2,3]
morphologyExKszieArgList = [2,3,4,5]

rank = 0
if not os.path.exists(imname):
    os.system("mkdir " + imname)


for x in tqdm(itertools.product(blurMethodList,
                           blurKsizeArgList,
                           thresholdMethodList,
                           thresholdArgList,
                           thresholdBlocksizeArgList,
                           thresholdCArgList,
                           morphologyExMethodList,
                           morphologyExKszieArgList)):

    img = grayImg
    
    blurMethod = x[0]
    blurKsizeArg = x[1]

    thresholdMethod = x[2]
    thresholdArg = x[3]
    thresholdBlocksizeArg = x[4]
    thresholdCArg = x[5]

    morphologyExMethod = x[6]
    morphologyExKszieArg = x[7]

    #去噪
    ksize = (blurKsizeArg,blurKsizeArg)
    if blurMethod == 0: 
        #均值滤波
        img = cv2.blur(img,ksize)
    elif blurMethod == 1:
        #高斯滤波
        img = cv2.GaussianBlur(img,ksize,0)
    elif blurMethod == 2:
        #中值滤波
        img = cv2.medianBlur(img,blurKsizeArg)
    # elif blurMethod == 3:
    #     #双边滤波
    #     img = cv2.bilateralFilter(img,ksize)
    else:
        print("blurMethod ERROR")
        exit(1)


    #二值化处理 需要反过来
    threshold = thresholdArg
    blocksize = thresholdBlocksizeArg
    C = thresholdCArg
    if thresholdMethod == 0:
        #固定阈值二值化
        mask, img = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    elif thresholdMethod == 1:
        #自适应阈值二值化
            #三角法
            mask, img = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)
    elif thresholdMethod == 2:
            #OTSU
            mask, img = cv2.threshold(img,threshold,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif thresholdMethod == 3:
        #区域阈值二值化
            #区域均值
            img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,blocksize,C)
    elif thresholdMethod == 4:
            #区域高斯
            img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blocksize,C)
    else:
        print("thresholdMethod ERROR")
        exit(2)

    #形态学运算
    kernel = np.ones((morphologyExKszieArg,morphologyExKszieArg),np.uint8)
    if morphologyExMethod == 0:
        pass
    elif morphologyExMethod == 1:
        img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    elif morphologyExMethod == 2:
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    elif morphologyExMethod == 3:
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
        img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    else:
        print("morphologyExMethod ERROR")
        exit(3)

    if rank == 6537:
        a = 1

    char_pixel_num = 0
    for pixel in img[0,1:]:
        if pixel != 0:
            char_pixel_num += 1
    for pixel in img[-1,:-2]:
        if pixel != 0:
            char_pixel_num += 1
    for pixel in img[:-2,0]:
        if pixel != 0:
            char_pixel_num += 1
    for pixel in img[1:,-1]:
        if pixel != 0:
            char_pixel_num += 1
    if char_pixel_num > img.shape[0]+img.shape[1] - 2:
        img = 255 - img
    R,G,B = cv2.split(originImg)
    img = cv2.merge([R&img, G&img ,B&img])

    cv2.imwrite("./"+imname+"/"+imname+"_"+str(rank)+".jpg",img)
    rank = rank+1
