#coding:utf-8
import os
import cv2 as cv
import os
import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split


def visualize(output_dir,imgs, seq, seq_ord, seq_inv, seq_gt, enter, esc, length, epoch):
    """
    visualize a output of validation epoch
    
    :param output_dir: a path, which will be created when not exists
    :param imgs: torch GPU Tensor of (b,c,w,h)
    :param seq: torch GPU Tensor of (b,max_len,2)
    :param enter: (b,2)
    :param esc: (b,2)
    :param length: (b,1)
    """

    imgs = imgs.cpu()
    seq = seq.cpu()
    enter = enter.cpu()
    esc = esc.cpu()

    output_path = output_dir
    save_dir = output_path+'epoch_'+str(epoch)+'/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    for k in range((len(imgs))):
        img = imgs[k].numpy()
        c,w,h = img.shape
        img[0] = img[0]*0.229+0.485
        img[1] = img[1]*0.224+0.456
        img[2] = img[2]*0.225+0.406
        img = img * 255
        img.astype('int')
        img = img.transpose(1,2,0) # chw -> hwc
        img = img[...,::-1] #rgb --> bgr
        #print(enter[k],esc[k])
        img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_CONSTANT,value=[225,225,225])
        
        if seq_ord is not None:
            for j in range(length[k]):
                m, n = int(h/2+seq_ord[k][j,1]*(h/2-1)),int(h/2+seq_ord[k][j,0]*(h/2-1))
                if seq_ord[k][j,1] == 3:
                    break
                img[-m-3:-m+3,n-3:n+3,:] = np.zeros_like(img[-m-3:-m+3,n-3:n+3,:])
                img[-m-3:-m+3,n-3:n+3,0] = 100
                img[-m-3:-m+3,n-3:n+3,1] = 100
                img[-m-3:-m+3,n-3:n+3,2] = 200
        
        if seq_inv is not None:
            for j in range(length[k]):
                m, n = int(h/2+seq_inv[k][j,1]*(h/2-1)),int(h/2+seq_inv[k][j,0]*(h/2-1))
                if seq_inv[k][j,1] == 3:
                    break
                img[-m-3:-m+3,n-3:n+3,:] = np.zeros_like(img[-m-3:-m+3,n-3:n+3,:])
                img[-m-3:-m+3,n-3:n+3,0] = 200
                img[-m-3:-m+3,n-3:n+3,1] = 100
                img[-m-3:-m+3,n-3:n+3,2] = 100
        
        for j in range(length[k]):
            m, n = int(h/2+seq[k][j,1]*(h/2-1)),int(h/2+seq[k][j,0]*(h/2-1))
            if seq[k][j,1] == 3:
                break
            img[-m-3:-m+3,n-3:n+3,:] = np.zeros_like(img[-m-3:-m+3,n-3:n+3,:])
            #print(img[:,int(seq[k][j,1]*h)+256,256+int(seq[k][j,0]*h)])
        
        if seq_gt is not None:
            for j in range(1,length[k]-1): # omit start point
                m, n = int(h/2+seq_gt[k][j,1]*(h/2-1)),int(h/2+seq_gt[k][j,0]*(h/2-1))
                if seq[k][j,1] == 3:
                    break
                img[-m-3:-m+3,n-3:n+3,:] = np.zeros_like(img[-m-3:-m+3,n-3:n+3,:]) + 100
        
        enter = np.clip(enter,-0.95,1.)
        esc = np.clip(esc,-0.95,1.)
        
        # 红色是入点
        img[-int(h/2+enter[k][1]*(h/2-1)):-int(h/2+enter[k][1]*(h/2-1))+6,int(h/2+enter[k][0]*(h/2-1))-6:int(h/2+enter[k][0]*(h/2-1)),0] = 0
        img[-int(h/2+enter[k][1]*(h/2-1)):-int(h/2+enter[k][1]*(h/2-1))+6,int(h/2+enter[k][0]*(h/2-1))-6:int(h/2+enter[k][0]*(h/2-1)),1] = 0
        img[-int(h/2+enter[k][1]*(h/2-1)):-int(h/2+enter[k][1]*(h/2-1))+6,int(h/2+enter[k][0]*(h/2-1))-6:int(h/2+enter[k][0]*(h/2-1)),2] = 200
        
        # 蓝色是出点
        img[-int(h/2+esc[k][1]*(h/2-1)):-int(h/2+esc[k][1]*(h/2-1))+6,int(h/2+esc[k][0]*(h/2-1))-6:int(h/2+esc[k][0]*(h/2-1)),0] = 200
        img[-int(h/2+esc[k][1]*(h/2-1)):-int(h/2+esc[k][1]*(h/2-1))+6,int(h/2+esc[k][0]*(h/2-1))-6:int(h/2+esc[k][0]*(h/2-1)),1] = 0
        img[-int(h/2+esc[k][1]*(h/2-1)):-int(h/2+esc[k][1]*(h/2-1))+6,int(h/2+esc[k][0]*(h/2-1))-6:int(h/2+esc[k][0]*(h/2-1)),2] = 0
        
        #蓝色 出点方向
        
        #if esc[k][0] == 1:
        #    img[-5:,:,0] = 200
        #    img[-5:,:,1] = 0
        #    img[-5:,:,2] = 0
        #elif esc[k][1] == 1:
        #    img[:,:5,0] = 200
        #    img[:,:5,1] = 0
        #    img[:,:5,2] = 0
        #elif esc[k][2] == 1:
        #    img[:5,:,0] = 200
        #    img[:5,:,1] = 0
        #    img[:5,:,2] = 0
        #elif esc[k][3] == 1:
        #    img[:,-5:,0] = 200
        #    img[:,-5:,1] = 0
        #    img[:,-5:,2] = 0
        #print(str(i*32+k)+'.png')
        
        cv.imwrite(os.path.join(save_dir,str(k)+'_newdata.png'),img)

