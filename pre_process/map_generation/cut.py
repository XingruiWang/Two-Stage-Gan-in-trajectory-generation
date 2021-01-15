import cv2 as cv
import os
import numpy as np

dir = 'output\\'
#334 442


dx = 473
dy = 63
num=0
for i in os.listdir(dir):
    if i[-3:]!='png':
        continue
    pict = cv.imread(os.path.join(dir,i))
    pict = pict[dy:-dy,dx:-dx,:]
    cv.imencode('.png', pict)[1].tofile('output_cut/%s.png'%('0'*(3-len(str(num)))+str(num)))
    num+=1



