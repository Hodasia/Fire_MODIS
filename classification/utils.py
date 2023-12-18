import numpy as np
import os
import random
import sys,os
import torch
path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
import data_config
dc=data_config.DataConfig

##采样训练集
def sample_train(batch_num_pos, bathc_num_neg):
    bacth_data=[]
    # for _ in range(batch_num):
    #     pos_id=random.randint(1, dc.train_pos_num-1)
    #     pos_data=np.load('/workplace/dataset/MODIS/train/positive/'+str(pos_id)+'.npy')#64*7*15*15

    #     neg_id=random.randint(1, dc.train_neg_num-1)
    #     neg_data=np.load('/workplace/dataset/MODIS/train/negative/'+str(neg_id)+'.npy')#64*7*15*15

    #     bacth_data.extend(pos_data)
    #     bacth_data.extend(neg_data)
    for _ in range(batch_num_pos):
        pos_id=random.randint(1, dc.train_pos_num)
        pos_data=np.load('/workplace/dataset/MODIS_new/train_80/positive/'+str(pos_id)+'.npy')#64*7*15*15
        bacth_data.extend(pos_data)

    for _ in range(bathc_num_neg):
            neg_id=random.randint(1, dc.train_neg_num)
            neg_data=np.load('/workplace/dataset/MODIS_new/train_80/negative/'+str(neg_id)+'.npy')#64*7*15*15
            bacth_data.extend(neg_data)

    bacth_data=np.asarray(bacth_data)#256*11*15*15

    batch_x=bacth_data[:,:9]#256*9*15*15
    batch_y=bacth_data[:,-1]#256*1*15*15
    batch_y[batch_y<80]=0
    batch_y[batch_y>=80]=1
    batch_y=np.sum(batch_y,axis=(-1,-2))#256*1
    batch_y[batch_y > 0]=1
    batch_y = batch_y.reshape(-1, 1)

    return batch_x,batch_y

##采样验证集或测试集
def sample_val_test(img_id, type):
    data = np.load(os.path.join('/workplace/dataset/MODIS_new/', type+'_80/'+str(img_id)+'.npy'))#11*2030*1354

    batch_x=data[:,:9]#12150*9*15*15
    batch_y=data[:,-1]#12150*1*15*15
    batch_y[batch_y<80]=0
    batch_y[batch_y>=80]=1
    batch_y=np.sum(batch_y,axis=(-1,-2))#12150*1
    batch_y[batch_y > 0]=1
    batch_y = batch_y.reshape(-1, 1)

    return batch_x,batch_y

## 归一化
def nor2(x):
    b, _, h, w = x.shape
    num_channels = x.shape[1]

    x_new = np.zeros((b, num_channels, h, w))

    for i in range(num_channels):
        min_val = getattr(dc, f"min{i}")
        max_val = getattr(dc, f"max{i}")
        x_new[:, i] = (x[:, i] - min_val) / (max_val - min_val)

    return x_new

## 统计tp, fp, fn
def cal_hit(pred,true):
    #0/1:b,1

    # print(pred.flatten().tolist().count(0),pred.flatten().tolist().count(1))
    # print(true.flatten().tolist().count(0), true.flatten().tolist().count(1))

    ##tp:p+t=2
    tp=pred+true
    tp=tp.flatten().tolist().count(2)

    ##fp:p-t=1
    fp=pred-true
    fp = fp.flatten().tolist().count(1)

    ##fn:p-t=-1
    fn = pred-true
    fn = fn.flatten().tolist().count(-1)

    return tp,fp,fn

## 评估模型
def eval(args, model, type):
    model.eval()
    tp=0
    fp=0
    fn=0

    index=1
    num=args.test_epochs
    thrd=args.evaluate_threshold
    with torch.no_grad():
        while(index<=num):
            # print(index)
            test_x, test_y = sample_val_test(index, type)

            ims = nor2(test_x)  
            input = torch.Tensor(ims).cuda()
            pred = model(input)  # 8*9*101*101*1
            pred=pred.detach().cpu().numpy()
            pred[pred<=thrd]=0
            pred[pred>thrd]=1

            tpp,fpp,fnn= cal_hit(pred,test_y)
            tp+=tpp
            fp+=fpp
            fn+=fnn

            index += 1

    return tp,fp,fn