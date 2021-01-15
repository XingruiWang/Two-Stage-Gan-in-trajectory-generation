#!/usr/bin/env python3
import math
import os
import numpy as np
import matplotlib.pyplot as plt

grid_data = '/path/to/grid/data'


length_list = []  # sequence length
for file in os.listdir(grid_data):
    if file[-3:] != 'npy':
        continue
    tmp = np.load(grid_data+file, allow_pickle=True)
    # count sequence length
    length_list.append(round(sum(tmp/15)))

# drop the outlier in the original dataset
length_list = [x for x in length_list if x < 700]

plt.hist(length_list, bins=50, density=True, alpha=0.7, rwidth=0.85)
plt.xlabel('Trajectory sequence length')
plt.ylabel('Frequency')
plt.title('Original dataset Length Distribution')
plt.savefig('distribution.pdf')
