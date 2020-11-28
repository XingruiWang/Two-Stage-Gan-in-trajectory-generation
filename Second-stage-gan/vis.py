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
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

'''
def visualize(output_dir,imgs, seq, enter, esc, length, epoch, seq_gt):
    """
    visualize a output of validation epoch

    :param output_dir: a path, which will be created when not exists
    :param imgs: torch GPU Tensor of (b,c,w,h)
    :param seq: torch GPU Tensor of (b,max_len,2)
    :param enter: (b,2)
    :param esc: (b,2)
    :param length: (b,1)
    """
    direction = {0:'below', 1:'left', 2:'up', 3:'right'}

    imgs = imgs.cpu()
    seq = seq.cpu()
    enter = enter.cpu()
    esc = esc.cpu()
    esc_d = esc.nonzero().tolist() # (b,4)

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
        img = img.astype('uint8')
        img = img.transpose(1,2,0) # chw -> hwc
        print(img.shape)

        seq_transfer = [(int(h/2+p[0]*(h/2-1)), 512-int(h/2+p[1]*(h/2-1))) for p in seq[k] if p[1] != 3]
        enter_transfer = (int(h/2+enter[k][0]*(h/2-1)),512-int(h/2+enter[k][1]*(h/2-1)))
        exit_transfer = (int(h/2+esc[k][0]*(h/2-1)),512-int(h/2+esc[k][1]*(h/2-1)))

        seq_transfer = [enter_transfer] + seq_transfer
        seq_transfer = seq_transfer + [exit_transfer] if exit_transfer != (256,256) else seq_transfer
        print(seq_transfer)
        img = Image.fromarray(img)
        canvas = ImageDraw.Draw(img)
        canvas.line(seq_transfer, fill=(66,95,156,255), width=3)
        del canvas
        di = direction[esc_d[k][1]]
        img.save(os.path.join(save_dir,str(k)+'di-{}'.format(di)+'.png'), 'PNG')



'''
def visualize(output_dir,imgs, seq, enter, esc, length, epoch, seq_gt):


    """
    visualize a output of validation epoch

    :param output_dir: a path, which will be created when not exists
    :param imgs: torch GPU Tensor of (b,c,w,h)
    :param seq: torch GPU Tensor of (b,max_len,2)
    :param enter: (b,2)
    :param esc: (b,4)
    :param length: (b,1)
    """

    imgs = imgs.cpu()
    seq = seq.cpu()
    seq_gt = seq_gt.cpu().numpy()
    enter = enter.cpu()
    esc = esc.cpu()
    esc_d = esc.nonzero().tolist() # (b,4)

    direction = {0:'below', 1:'left', 2:'up', 3:'right'}
    output_path = output_dir
    save_dir = output_path+'epoch_'+str(epoch)+'/' if epoch != 'pred' else output_path + '/'
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
        gt_length = len(seq_gt[k])
        img = cv.copyMakeBorder(img, 5, 5, 5, 5, cv.BORDER_CONSTANT,value=[225,225,225])
        for j in range(length[k]):
            m, n = int(h/2+seq[k][j,1]*(h/2-1)),int(h/2+seq[k][j,0]*(h/2-1))
            if seq[k][j,1] == 3:
                break
            img[-m-3:-m+3,n-3:n+3,:] = np.zeros_like(img[-m-3:-m+3,n-3:n+3,:])
            #print(img[:,int(seq[k][j,1]*h)+256,256+int(seq[k][j,0]*h)])

        for j in range(gt_length):
            m, n = int(h/2+seq_gt[k][j,1]*(h/2-1)),int(h/2+seq_gt[k][j,0]*(h/2-1))
            if seq_gt[k][j,1] == 3:
                break
            img[-m-3:-m+3,n-3:n+3,:] = np.full_like(img[-m-3:-m+3,n-3:n+3,:], 150)

        # 红色是入点
        img[-int(h/2+enter[k][1]*(h/2-1))-3:-int(h/2+enter[k][1]*(h/2-1))+3,int(h/2+enter[k][0]*(h/2-1))-3:int(h/2+enter[k][0]*(h/2-1))+3,0] = 0
        img[-int(h/2+enter[k][1]*(h/2-1))-3:-int(h/2+enter[k][1]*(h/2-1))+3,int(h/2+enter[k][0]*(h/2-1))-3:int(h/2+enter[k][0]*(h/2-1))+3,1] = 0
        img[-int(h/2+enter[k][1]*(h/2-1))-3:-int(h/2+enter[k][1]*(h/2-1))+3,int(h/2+enter[k][0]*(h/2-1))-3:int(h/2+enter[k][0]*(h/2-1))+3,2] = 200

        #蓝色 出点方向
        if esc[k][0] == 1:
            img[-5:,:,0] = 200
            img[-5:,:,1] = 0
            img[-5:,:,2] = 0
        elif esc[k][1] == 1:
            img[:,:5,0] = 200
            img[:,:5,1] = 0
            img[:,:5,2] = 0
        elif esc[k][2] == 1:
            img[:5,:,0] = 200
            img[:5,:,1] = 0
            img[:5,:,2] = 0
        elif esc[k][3] == 1:
            img[:,-5:,0] = 200
            img[:,-5:,1] = 0
            img[:,-5:,2] = 0
        #print(str(i*32+k)+'.png')

        di = direction[esc_d[k][1]]

        cv.imwrite(os.path.join(save_dir,str(k)+'di-{}'.format(di)+'.png'),img)

