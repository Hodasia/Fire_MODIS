import numpy as np
import os
from tqdm import trange

# img = np.load('/workplace/dataset/MODIS/valid/'+str(1)+'.npy')
# print('**************')
# print(img[0, 0])
# print(np.max(img[0,0]))

# find_min = []
# for pos_id in trange(1, 4484):
#     pos_data = np.load('/workplace/dataset/MODIS/train/positive/'+str(pos_id)+'.npy')
#     for i in range(64):
#         img_max = np.max(pos_data[i, 1])
#         find_min = np.append(find_min, img_max, axis=None)
# print(find_min)
# print(np.max(find_min)) # 96.91171264648438
# print(np.min(find_min)) # 0.5110194087028503

def sample_val(idx):
    file_idx = (idx // 12150) + 1 ## npy文件索引
    data=np.load('/workplace/dataset/MODIS/valid/'+str(file_idx)+'.npy')#12150*7*15*15
    img_idx = idx - 12150 * (file_idx - 1) ## 子图索引

    offset = 0 # 筛选子图的偏移量
    num_filtered = 0
    batch_num = 0
    batch_data = []
    
    while(img_idx<12150):
        offset += 1

        ## 根据阈值筛选子图
        if np.max(data[img_idx, 1]) < 1.1:
            num_filtered += 1
            continue

        # 一个batch有12150张子图
        img = data[img_idx][np.newaxis, :]
        batch_data.extend(img)
        batch_num += 1
        if batch_num == 12150:
            break

        # 读完一个npy文件, 加载新的npy文件
        if (img_idx + 1) == 12150:
            file_idx += 1
            data=np.load('/workplace/dataset/MODIS/valid/'+str(file_idx)+'.npy')
            img_idx = 0
        else:
            img_idx += 1

    batch_data = np.asarray(batch_data) # 12150*6*15*15
    batch_x=batch_data[:,:6]#12150*6*15*15
    batch_y=batch_data[:,6:]#12150*1*15*15
    batch_y[batch_y<9]=0
    batch_y[batch_y==9]=1
    batch_y=np.sum(batch_y,axis=(-1,-2))#12150*1

    return batch_x,batch_y, offset, num_filtered, batch_num

idx = 0
while(idx < (12150*2)):
    batch_x,batch_y, offset, num_filtered, batch_num = sample_val(idx)
    print(num_filtered)
    print(batch_num)
    idx += offset
    print(idx)
print(batch_x.shape, batch_y.shape, offset)
