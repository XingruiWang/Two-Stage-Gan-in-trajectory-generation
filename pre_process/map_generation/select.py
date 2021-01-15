import cv2 as cv
import os
import numpy as np

def min_pooling(img, G=2):
    print('pooling')
    H, W, C = img.shape
    out = np.full(img.shape,0)

    Nh = int(H/G)
    Nw = int(W/G)

    for j in range(Nh):
        for i in range(Nw):
            for c in range(C):
                out[G*j:G*(j+1), G*i:G*(i+1),c] = np.min(img[G*j:G*(j+1), G*i:G*(i+1),c]).astype(int)
    return out

def same(p1,p2):
    return (min_pooling(np.abs(p1-p2))<50).all()
dir = 'test//'

pict_list = os.listdir(dir)

start_pict_name = pict_list[0]
start_pict = min_pooling(cv.imread(os.path.join(dir,start_pict_name))).astype(int)
selected = [[start_pict_name]]
cp1 = start_pict
cp2 = start_pict
is_start = False
for pic_name in pict_list[1:30]:
    pic = min_pooling(cv.imread(os.path.join(dir,pic_name))).astype(int)
    c = min_pooling(np.abs(cp1[0:63,:,:]-pic[-126:-63,:,:]))
    cv.imencode('.png', c)[1].tofile(pic_name)
    #cv.imencode('.png', cp1[0:63,:,:])[1].tofile('test.png')
    #cv.imencode('.png', pic[-126:-63,:,:])[1].tofile('test2.png')
    #print(pic_name,(average_pooling(cv.subtract(cp1[0:63,:],pic[-126:-63,:]))<20).all())
    print(pic_name,c.max())
    if is_start or same(cp1[0:63,:,:],pic[-126:-63,:,:]):
        is_start = False
        if len(selected[0])<10:
            cp1 = pic[:,:,:]
            selected[-1].append(pic_name)
            if len(selected[-1])==10:
                selected.append([])
                is_start = True
        else:
            if (cv.subtract(cp2[:,-473:,:],pic[:,473:946,:])<20).all():
                cp2_name = selected[-2][len(selected[-1])]
                cp2 = cv.imread(os.path.join(dir,cp2_name),cv.IMREAD_GRAYSCALE)
                selected[-1].append(pic_name)
                if len(selected[-1])==10:
                    selected.append([])
                    is_start = True

print(selected)







