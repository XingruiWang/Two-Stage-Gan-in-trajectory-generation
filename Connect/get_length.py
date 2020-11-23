#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:12:55 2020

@author: meteor
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt


#ori_data = '/Users/meteor/Desktop/论文/output_inter/'
gen_data = '/home/wxr/projects/gan/output_generated'

#EARTH_REDIUS = 6378.137
#
#
#def rad(d):
#    return d * math.pi / 180.0
#
#
#def getDistance(lat1, lng1, lat2, lng2):
#    radLat1 = rad(lat1)
#    radLat2 = rad(lat2)
#    a = radLat1 - radLat2
#    b = rad(lng1) - rad(lng2)
#    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
#    s = s * EARTH_REDIUS
#    return s

#def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
#    """
#    Calculate the great circle distance between two points
#    on the earth (specified in decimal degrees)
#    """
#    # 将十进制度数转化为弧度
#    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#
#    # haversine公式
#    dlon = lon2 - lon1
#    dlat = lat2 - lat1
#    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#    c = 2 * asin(sqrt(a))
#    r = 6371 # 地球平均半径，单位为公里
#    return c * r * 1000

length_list = []
for file in os.listdir(gen_data):
    if file[-3:] != 'npy':
        continue
    tmp = np.load(gen_data+file, allow_pickle = True)
    length_list.append(round(sum(tmp/15)))
#    length = 0
#
#    for p in range(len(tmp)-1):
#        d = getDistance(tmp[p][0], tmp[p][1], tmp[p+1][0], tmp[p+1][1])
#        print(d)
#        length += d

#    length_list.append(length)

#
length_list = [x for x in length_list if x < 700]
plt.hist(length_list, bins =50, density=True, alpha=0.7, rwidth=0.85)
plt.xlabel('Trajectory Length in kilometers')
plt.ylabel('Frequency')
plt.title('Original dataset Length Distribution')

