import numpy as np
import os
from collections import Counter
from tqdm import tqdm
import random

## 切割为15x15的子图
def chunk_list(h,w,ks,skip):
    hi,wi=0,0
    chunk_lisk=[]
    while(hi+ks<h):
        if(wi+ks<w):
            tmp=[hi,hi+ks,wi,wi+ks]
            chunk_lisk.append(tmp)
            wi+=skip
        else:
            wi=0
            hi+=skip
    return chunk_lisk

path = '/workplace/dataset/MODIS_new/2021_new/'
path_parent = '/workplace/dataset/MODIS_new/train_60/'
if not os.path.exists(path_parent):
    os.makedirs(path_parent)
path_train_pos = os.path.join(path_parent, 'positive')
if not(os.path.exists(path_train_pos)):
    os.mkdir(path_train_pos)
path_train_neg = os.path.join(path_parent, 'negative')
if not(os.path.exists(path_train_neg)):
    os.mkdir(path_train_neg)

all = os.listdir(path)
all.sort()
print(len(all))
ks=15
skip=ks
h,w=2030,1354
cl=chunk_list(h,w,ks,skip)

## total image counter
num_total, num_pos, num_neg, num_zero, num_filter, num_error = 0, 0, 0, 0, 0, 0
## batch counter
cnt_neg, cnt_pos = 0, 0
## file name counter
name_pos, name_neg = 0, 0
pos = []
neg = []

batch_num = 64
con_thrd, c21_thrd, c22_thrd=60, 0.5, 0.79
filter_ratio = 1/15

for a in all:
    data = np.load(os.path.join(path, a))['arr_0']
    num_error = 0

    for c in cl:
        d=data[:, c[0]:c[1],c[2]:c[3]] #(11, 15, 15)
        # num_total += 1
        
        confidence=d[-1]

        if confidence.size > 0:
            max_con=np.max(confidence)
        else:
            num_error += 1
            path_error = os.path.join(path, a)
            print(5 * '#' + f'\nERROR! Confidence array size is equal to 0 in: {path_error} for {num_error} times\n' + 5 * '#')
            continue

        num_total += 1

        # pos
        if max_con >= con_thrd:
            num_pos += 1
            cnt_pos += 1
            pos.append(d)
            if cnt_pos == batch_num:
                cnt_pos = 0
                name_pos += 1
                file_name_pos = str(name_pos) + '.npy'
                pos = np.asarray(pos)
                np.save(os.path.join(path_train_pos, file_name_pos), pos)
                pos = []
        # filter
        elif max_con == 0:
            c21=np.max(d[4])
            c22=np.max(d[5])
            if (not (c21 > c21_thrd and c22 > c22_thrd)) and (random.random() <= filter_ratio):
                num_zero += 1
                cnt_neg += 1
                neg.append(d)
                if cnt_neg == batch_num:
                    cnt_neg = 0
                    name_neg += 1
                    file_name_neg = str(name_neg) + '.npy'
                    neg = np.asarray(neg)
                    np.save(os.path.join(path_train_neg, file_name_neg), neg)
                    neg=[]            
                
            else:
                num_filter += 1
                continue
            # num_filter += 1
        # neg
        else:
            num_neg += 1
            cnt_neg += 1
            neg.append(d)
            if cnt_neg == batch_num:
                cnt_neg = 0
                name_neg += 1
                file_name_neg = str(name_neg) + '.npy'
                neg = np.asarray(neg)
                np.save(os.path.join(path_train_neg, file_name_neg), neg)
                neg=[]
        # print(max_con)

    # print(num_total, num_pos, num_neg, num_filter)
    print(num_total, num_pos, num_neg, num_zero, num_filter)


print('###########################')
print('Total sub img:', num_total)
print('Positive: {:d}, Negative: {:d}, Negative Zero: {:d}, Filtered Neg: {:d}'.format(num_pos, num_neg, num_zero, num_filter))

## 错误版本
## Error: 270 /workplace/dataset/MODIS_new/2021_new/20212271150.npz

# Total sub img: 31,407,750
# Positive: 653,743 = 10,214 * 64 + 47

## Con_thrd=60, c21=0.7, c22=0.8
#  Negative: 13,721,687 = 214401 * 64 + 23 , Filtered Neg: 17,032,050 (54%), pos:neg=1:21

## Con_thrd=60, c21=0.8, c22=0.8
#  Negative: 13,690,308 = 213,911 * 64 + 4, Filtered Neg: 17,063,429 (54%), pos:neg=1:21

## Con_thrd=60, c21=0.8, c22=0.9
#  Negative: 11,351,549 = 177,367 * 64 + 67, Filtered Neg: 19,402,188 (61%), pos:neg=1:17

## Con_thrd=60, c21=0.9, c22=1.0
#  Negative: 9,776,520 = 152,758 * 64 + 8, Filtered Neg: 20,977,217 (67%), pos:neg=1:15

## Con_thrd=60 c21=1.0, c22=1.1
#  Negative: 8,455,952 = 132,124 * 64 + 16, Filtered Neg: 22,297,785 (71%), pos:neg=1:13

## Con_thrd=60, c21=1.1, c22=1.2
#  Negative: 7,266,294 = 113,535 * 64 + 54 , Filtered Neg: 23,487,443 (75%), pos:neg=1:11

#########################################################################
# Con_thrd=60，c21=0.5, c22=0.79
# Total sub img: 31,407,480 + 270
# Positive: 653,743, Negative: 189,701, Negative Zero: 16,729,380, Filtered Neg: 13,834,656 (44%), pos:neg=1:26

# c21=0.4, c22=0.7
# Total sub img: 31,407,480 + 270
# Positive: 653,743, Negative: 189,701, Negative Zero: 13,456,258, Filtered Neg: 17,107,778 (54%), pos:neg=1:21

# Con_thrd=60，c21=0.5, c22=0.79, filter_ratio=1/15
# Total sub img: 31,407,480 + 270
# Positive: 653,743, Negative: 189,701, Negative Zero: 1,116,594, Filtered Neg: 29,447,442 (94%), pos:neg=1:2

#########################################################################
# Con_thrd=80
# Total sub img: 31,407,480 + 270
# Positive: 310,001, Negative: 533,443, Filtered Neg: 30,564,036 (97%), pos:neg=1:1.7

# c21: 0.5, c22: 0.79
# Total sub img: 31,407,480
# Positive: 310,001, Negative: 533,443, Negative Zero: 16,729,380, Filtered Neg: 13,834,656 (43%), pos:neg=1:56