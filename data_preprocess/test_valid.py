import numpy as np
import os

def find_thrd(cnt, path, data, c21, c22, flag):
    num = data.shape[0]
    # num_error = 0
    thrd = 80
    pos, neg, filter, tn, fn = 0, 0, 0, 0, 0
    img = []
    file_name = str(cnt) + '.npy'
    parent_path = '/workplace/dataset/MODIS_new/'

    for i in range(num):
        confidence = data[i, -1]
        p_c21, p_c22 = np.max(data[i, 4]), np.max(data[i, 5])

        # if confidence.size == 0:
        #     num_error += 1
        #     print(5 * '#' + f'\nERROR! Confidence array size is equal to 0 in: {path} for {num_error} times\n' + 5 * '#')
        #     continue

        max_con = np.max(confidence)

        if max_con >= thrd:
            pos += 1
            label = 1
        else:
            neg += 1
            label = 0
        
        if not (p_c21 >= c21 and p_c22 >= c22):
            filter += 1
            if label == 0: tn += 1
            elif label == 1: fn += 1
        else:
            img.append(data[i])

    img = np.asarray(img)
    np.save(os.path.join(parent_path, flag, file_name), img)
    return num, pos, neg, filter, tn, fn

##切分2022年的数据为15*15的子图，batch为12150，作为验证集和测试集
def cut_test(cnt, img_path, c21, c22, flag):
    # global cnt1, cnt2
    data=np.load(img_path)['arr_0']#11*2030*1354
    data=data[:,:2025,:1350]#11*2030*1354,保证宽和高被15整除
    c,h,w=data.shape
    data=data.reshape(c,(h//15)*(w//15),15,15)#11*12150*15*15
    batch_data=np.swapaxes(data,0,1)#12150*11*15*15

    ## 阈值筛选
    num,pos_num,neg,filter,tn,fn = find_thrd(cnt, img_path, batch_data, c21, c22, flag)
    
    return num,pos_num,neg,filter,tn,fn

pathset = '/workplace/dataset/MODIS_new/2022_new/'
# test_path = '/workplace/dataset/MODIS_new/test_new'
test_path = '/workplace/dataset/MODIS_new/test_80'
if not os.path.exists(test_path):
    os.makedirs(test_path)
# valid_path = '/workplace/dataset/MODIS_new/valid_new'
valid_path = '/workplace/dataset/MODIS_new/valid_80'
if not os.path.exists(valid_path):
    os.makedirs(valid_path)

all_img = os.listdir(pathset)
thrd_arr = [[1.0, 1.1]]
cnt = 0
cnt_valid = 0
cnt_test = 0

for c21, c22 in thrd_arr:
    print("#####################")
    print("c21 thrd:{:g}, c22 thrd:{:g}".format(c21, c22))
    num_final, pos_final,neg_final, filter_final, tn_final, fn_final=0,0,0,0,0,0
    for img in all_img:
        img_path = os.path.join(pathset, img)
        # num,pos,neg,filter,tn,fn = cut_test(cnt_valid, img_path, c21, c22, 'valid')

        cnt += 1
        if cnt % 2 == 0: 
            flag= 'valid_80'
            cnt_valid += 1
            num,pos,neg,filter,tn,fn = cut_test(cnt_valid, img_path, c21, c22, flag)
        else: 
            flag = 'test_80'
            cnt_test += 1
            num,pos,neg,filter,tn,fn = cut_test(cnt_test, img_path, c21, c22, flag)
        
        num_final += num
        pos_final += pos
        neg_final += neg
        filter_final += filter
        tn_final += tn
        fn_final += fn
        print(f'total num: {num_final}, pos num: {pos_final}, neg_num: {neg_final}, filter_num: {filter_final}, tn: {tn_final}, fn:{fn_final}')

##########################################
## Con_thrd=60
# total num: 26,098,200, pos num: 792,039, neg_num 25,306,161
# c21:1.1, c22:0.8
# filter_num: 15,377,818, tn: 15,358,537(59%), fn:19,281(fn/total: 0.074%, fn/pos: 2.4%)
# c21:1.0, c22:0.8
# filter_num: 13,973,806, tn: 13,963,664(54%), fn:10,142(fn/total: 0.038%, fn/pos: 1.2%)
# c21:0.9, c22:0.8
# filter_num: 12,248,700, tn: 12,244,214(47%), fn:4,486(fn/total: 0.017%, fn/pos: 0.57%)
# c21:0.8, c22:0.8
# filter_num: 10,218,020, tn: 10,217,356(39%), fn:664(fn/total:0.0025% fn/pos: 0.084%)
# c21:0.85, c22:0.8
# filter_num: 11,142,260, tn: 11,140,045(43%), fn:2,215(fn/total:0.0085% fn/pos: 0.28%)

###########################################
## Con_thrd=80, c21:1.0, c22:1.1
# total num: 26,098,200, pos num: 371,045, neg_num: 25,727,155, filter_num: 15,355,571 (59%), tn: 15,354,051, fn:1,520 (fn/total:0.0058% fn/pos: 0.41%)