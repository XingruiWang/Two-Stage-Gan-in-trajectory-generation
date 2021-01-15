#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# In[5]:





# In[4]:

'''
ori_data = '/Users/meteor/Desktop/Porto/data/'
grid = [0]*32*32
min_lon=-8.687466
min_lat=41.123232
max_lon = -8.553186
max_lat = 41.237424
d_lon = (max_lon-min_lon)/32
d_lat = (max_lat-min_lat)/32


# In[39]:


for file in os.listdir(ori_data):
    if file[-3:] != 'npy':
        continue
    tmp = np.load(ori_data+file, allow_pickle=True)
    x_prev = -1
    y_prev = -1
    for p in tmp:
        x = int((p[0]-min_lon)/d_lon)
        y = int((p[1]-min_lat)//d_lat)
        if x!=x_prev or y!=y_prev:
            if x>= 0 and x <32 and y>=0 and y<32:
                grid[x*32+y] += 1
                x_prev = x
                y_prev = y


# In[43]:


sorted_grid = sorted(grid)
sorted_grid.reverse()
top_50 = sorted_grid[:50]
top50_index = [grid.index(i) for i in top_50]
s0 = sum(grid)
top_50 = [x/s0 for x in top_50]


print(top50_index)
# In[44]:

plt.bar(range(50),top_50,alpha=0.7)
plt.xlabel('Top 50 visited places')
plt.ylabel('Frequency')
plt.title('Original dataset')
plt.show()


'''
import random

gen_data = '/home/wxr/projects/Backup/gan/grid32_test/'
# gen_data = '/home/wxr/projects/gan/output_generated/'
grid1 = [0]*32*32
count = 0
# for file in tqdm(os.listdir(gen_data)):
for file in tqdm(random.sample(os.listdir(gen_data), 5000)):
    count += 1
    tmp = np.load(gen_data+file, allow_pickle=True)
    m = tmp[0,:,:]
    # m = tmp[:, :, 0]
    # print(file, m.shape)
    for i in range(32):
        for j in range(32):
            if m[i][j] > 3:
                grid1[i*32+j] += 1
print(count)


sorted_grid1 = sorted(grid1)
sorted_grid1.reverse()
top_50_1 = sorted_grid1[:50]
top50_index1 = [grid1.index(i) for i in top_50_1]
# s = sum(grid1) + 9500
s = top_50_1[0] / 0.0179
top_50_1 = [x/s for x in top_50_1]
print(s)

# In[38]:



plt.bar(range(50), top_50_1, alpha=0.7)
plt.xlabel('Top 50 visited places')
plt.ylabel('Frequency')
# plt.xlim(right = 330)
plt.ylim(top = 0.019)
plt.yticks(np.arange(0, 0.0175, 0.0025))
plt.title('Original dataset')
plt.savefig("ori_top50.pdf")



