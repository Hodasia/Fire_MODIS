import os
import numpy as np

# thrd=1.0
# test_all=os.listdir('/workplace/dataset/MODIS/test/')
# num_pos = 0
# pathset='/workplace/dataset/MODIS/test_v2/'
# if not(os.path.exists(pathset)):
#     os.mkdir(pathset)

# print('************** BEGIN FILTER *****************')
# for v in test_all:
#     new_data = []
#     data=np.load('/workplace/dataset/MODIS/test/'+v)#b,c,h,w
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
# print(num_pos) # 5968046

tn=0
fn=0
num=0
pos_num=0
pos_num_plus=0
other_num=0
filter_num=0

thrd=1.0
test_all=os.listdir('/workplace/dataset/MODIS/test/')
idx=0
for v in test_all:
    data=np.load('/workplace/dataset/MODIS/test/'+v)#b,c,h,w
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
                print(v, i, l) 
                print('****************************')
                other_num+=1

    idx+=1
    print(idx,tn,fn,num,pos_num, pos_num_plus)

print(num-filter_num)
print(other_num)

#num=13,049,100
#pos_num=116,985
#thrd=1.0:  tn=7,080,838(54.26%), fn=212(0.18%)
#thrd=1.05: tn=7,426,540(56.91%), fn=508(0.43%)
#thrd=1.1:  tn=7,765,016(59.50%), fn=916(0.78%)
#thrd=1.2:  tn=8,467,470(64.88%), fn=1,647(1.4%)