import numpy as np
import os

# thrd=1.0
# val_all=os.listdir('/workplace/dataset/MODIS/valid/')
# num_pos = 0
# pathset='/workplace/dataset/MODIS/valid_v2/'
# if not(os.path.exists(pathset)):
#     os.mkdir(pathset)

# print('************** BEGIN FILTER *****************')
# for v in val_all:
#     new_data = []
#     data=np.load('/workplace/dataset/MODIS/valid/'+v)#b,c,h,w
#     d=data[:,1]#b,h,w

#     for i in range(d.shape[0]):
#         p=np.max(d[i])
#         if p < thrd:
#             continue
#         else:
#             new_data.append(data[i])

#     new_data = np.asarray(new_data)
#     num_pos += new_data.shape[0]
#     print(new_data.shape, v)
#     np.save(os.path.join(pathset, v), new_data)
# print(num_pos) # 6151884


# valid_img = np.load('/workplace/dataset/MODIS/valid/'+str(1)+'.npy')
# print(np.max(valid_img[0, 1]))
# label = valid_img[0, 6]
# print(label)
# label[label < 9] = 0
# label[label == 9] = 1
# print(label)
# label = np.sum(label, axis=(-1, -2))
# print(label== 0)

# threshold = 0.5110194087028503

# num_filtered = 0 # 被当作负样本筛掉的总数
# num_fn = 0 # 被当作负样本筛掉，实际上却是正样本的数量
# num_fp = 0 # 未被筛掉的样本中负样本的数量

# for img_id in trange(1, 1075):
#     valid_img = np.load('/workplace/dataset/MODIS/valid/'+str(img_id)+'.npy') # 12150*7*15*15
#     for i in range(12150):
#         img_max = np.max(valid_img[i, 1])
#         label = valid_img[i, 6]
#         label[label < 9] = 0
#         label[label == 9] = 1
#         label = np.sum(label, axis=(-1, -2))
#         # 根据阈值筛掉负样本
#         if img_max < threshold:
#             num_filtered += 1
#             # 被筛掉的样本中是否有正样本
#             if label == 1:
#                 num_fn += 1
        
#         # 计数未被筛掉的样本中负样本的数量
#         else:
#             if label == 0:
#                 num_fp += 1

# print('****** valid ******')
# print('Total filtered negative data: {:d}'.format(num_filtered)) # 1015588
# print('Filtered positive data: {:d}'.format(num_fn)) # 0
# print('Unfiltered negative data: {:d}'.format(num_fp)) # 11843302

tn=0
fn=0
num=0
pos_num=0
pos_num_plus=0
filter_num = 0
other_num = 0

thrd=1.1
val_all=os.listdir('/workplace/dataset/MODIS/valid/')
idx=0
for v in val_all:
    data=np.load('/workplace/dataset/MODIS/valid/'+v)#b,c,h,w
    num+=data.shape[0]
    d=data[:,1]#b,h,w
    label=data[:,-1]#b,h,w
    label[label<9]=0
    label[label==9]=1
    label=np.sum(label,axis=(-1,-2))#b

    for i in range(d.shape[0]):
        p=np.max(d[i])
        l=label[i]
        if(l==1):pos_num+=1
        elif(l==2):pos_num_plus+=1

        if(p<thrd): 
            filter_num += 1
            if (l==0):tn+=1
            elif(l==1):fn+=1
            else: 
                print('****************************')
                print(v, i, l) # 819.npy 5402 2.0, 954.npy 6663 2.0, 302.npy 5822 2.0
                print('****************************')
                other_num+=1

    idx+=1
    print(idx,tn,fn,num,pos_num, pos_num_plus)

print(num-filter_num)#6151884
print(other_num)#3

#num=13,049,100
#pos_num=123,176 pos_num_plus = 49,270 total: 172,446
#thrd=0.5:  tn=947951,  fn=0
#thrd=1.0:  tn=6,897,087(52.8%), fn=126+3(0.1%)
#thrd=1.1:  tn=7,616,318(58.3%), fn=604(0.49%)
#thrd=1.2:  tn=8,329,731(63.8%), fn=1,158(0.9%)
#thrd=1.5:  tn=10,291,502,  fn=2,487(2%)
#thrd=2.0:  tn=12,228,940   fn=15,522(12.6%)
