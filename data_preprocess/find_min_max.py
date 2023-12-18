import os
import numpy as np
from tqdm import trange

path = '/workplace/dataset/MODIS_new/train_new/'
positive_all = os.listdir(path + 'positive')
negative_all = os.listdir(path + 'negative')

min_pos_list = [float('inf')] * 9
max_pos_list = [-float('inf')] * 9
min_neg_list = [float('inf')] * 9
max_neg_list = [-float('inf')] * 9

print('begin positive')
for j in range(len(positive_all)):
    pos_data = np.load('/workplace/dataset/MODIS_new/train_new/positive/' + str(j + 1) + '.npy')
    for i in range(9):
        # print(f'\nchannel**{i}**begins in positive dataset')
        min_pos_list[i] = min(min_pos_list[i], np.min(pos_data[:, i]))
        max_pos_list[i] = max(max_pos_list[i], np.max(pos_data[:, i]))

print('begin negative')
for k in range(len(negative_all)):
    neg_data = np.load('/workplace/dataset/MODIS_new/train_new/negative/' + str(k + 1) + '.npy')
    for i in range(9):
        # print(f'\nchannel**{i}**begins in negative dataset')
        min_neg_list[i] = min(min_neg_list[i], np.min(neg_data[:, i]))
        max_neg_list[i] = max(max_neg_list[i], np.max(neg_data[:, i]))

for i in range(9):
    min_final = min(min_pos_list[i], min_neg_list[i])
    max_final = max(max_pos_list[i], max_neg_list[i])
    print(f'min of channel{i}')
    print(min_final)
    print(f'max of channel{i}')
    print(max_final)

# min_pos = float ('inf')
# max_pos = -float ('inf')
# min_neg = float ('inf')
# max_neg = -float ('inf')
# min_pos_list = []
# max_pos_list = []
# min_neg_list = []
# max_neg_list = []
# # ('******BEGIN******')
# for j in range(len(positive_all)):
#     # print(j)
#     pos_data=np.load('/workplace/dataset/MODIS_new/train_new/positive/'+str(j+1)+'.npy')
#     for i in range(5, 10):
#         # print(i)
#         print('\n'+'channel**' + str(i) + '**begins in positive dataset')
#         min_pos = np.min(pos_data[:, i]) if np.min(pos_data[:, i]) < min_pos else min_pos
#         max_pos = np.max(pos_data[:, i]) if np.max(pos_data[:, i]) > max_pos else max_pos
#         min_pos_list[i-5] = min_pos
#         max_pos_list[i-5] = max_pos

# for k in range(len(negative_all)):
#     # print(k)
#     neg_data=np.load('/workplace/dataset/MODIS_new/train_new/negative/'+str(k+1)+'.npy')
#     for i in range(5, 10):
#         # print(i)
#         print('\n'+'channel**' + str(i) + '**begins in negative dataset')
#         min_neg = np.min(neg_data[:, i]) if np.min(neg_data[:, i]) < min_neg else min_neg
#         max_neg = np.max(neg_data[:, i]) if np.max(neg_data[:, i]) > max_neg else max_neg
#         min_neg_list[i-5] = min_neg
#         max_neg_list[i-5] = max_neg

# for i in range(5, 10):
#     min_final = min_pos_list[i-5] if min_pos_list[i-5] < min_neg_list[i-5] else min_neg_list[i-5]
#     max_final = max_pos_list[i-5] if max_pos_list[i-5] > max_neg_list[i-5] else max_neg_list[i-5]
#     print('min of channel'+ str(i))
#     print(min_final)
#     print('max of channel' + str(i))
#     print(max_final)
# print('******END******')
