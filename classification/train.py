import os
import argparse
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import random
import torch
import torch.nn as nn
import sys,os
import data_config
path = os.path.dirname(os.path.dirname(__file__)) 
sys.path.append(path)
import classification.model_resnet as m_resnet
# import classification.model_vit as m_vit
import utils
import logger
from datetime import datetime
import wandb

dc=data_config.DataConfig

def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds

def train():
    # start a new wandb run to track this script
    # wandb.login()
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="fire_resnet50_train",
    
    #     # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 1e-3,
    #     "epochs": 200,
    #     }
    # )

    set_seeds(args)
    best_val_perf = float('-inf')
    log = logger.Logger(args.model + '.log', on=True)
    log.log(str(args))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = m_resnet.resnet50()
    # model = m_vit.VisionTransformer()
    model.cuda()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.scheduler_gamma)

    log.log('***** train *****')
    log.log('# epochs: {:d}'.format(args.epochs))
    log.log('# batch size : {:d}'.format(args.sample))
    log.log('# learning rate: {:g}'.format(args.lr))
    log.log('# parameters: {:d}'.format(count_parameters(model)))
    log.log('***** valid *****')
    log.log('# valid epoch: {:d}'.format(args.test_epochs))
    log.log('# threshold: {:g}'.format(args.evaluate_threshold))

    epoch=1
    tr_loss, logging_loss = 0.0, 0.0
    step_num = 0
    current_best = 0.0
    curr_best_epoch = 0
    early_stop_cnt = 0
    early_stop_flag = False
    
    for i in range(1, args.sample * args.epochs + 1):
        if(i%args.sample==1):
            epoch_start_time = datetime.now()
            epoch_train_start_time = datetime.now()
        
        # 训练
        model.train()
        train_x,train_y= utils.sample_train(args.batch_num_pos, args.batch_num_neg)
        ims = utils.nor2(train_x)  # ims/255
        ims = torch.Tensor(ims).cuda()
        
        pred = model(ims) 
        loss = criterion(pred, torch.Tensor(train_y).cuda())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_num += 1
        tr_loss += loss.detach()

        # sample为一个epoch的采样次数，即batch的数量
        # 验证
        if(i%args.sample==0):

            log.log('training time for epoch {:3d} '
                   'is {:s}'.format(epoch, strtime(epoch_train_start_time)))
            
            scheduler.step()
            tp,fp,fn=utils.eval(args, model, 'valid')

            log.log('eval tp {:3d} | fp {:3d} | fn {:3d}'.format(tp, fp, fn))
            if(tp+fp>0 and tp+fn>0):
                p=tp/(tp+fp)
                r=tp/(tp+fn)
                if(p+r>0):
                    f1=2*p*r/(p+r)
                    fa=1-p
                    log.log('Done with epoch {:3d} | train loss {:8.4f} | '
                   'valid f1 {:8.4f}| valid precision {:8.4f}| valid recall {:8.4f} | valid fa {:8.4f}'
                   ' epoch time {} '.format(epoch, tr_loss / step_num, f1, p, r, fa, strtime(epoch_start_time)))
                    
                    # log metrics to wandb
                    # wandb.log({"valid f1": f1, "valid precision": p, "valid recall": r, "valid fa": fa})

                    if f1 > best_val_perf:
                        early_stop_cnt = 0
                        current_best = f1
                        curr_best_epoch = epoch
                        log.log('------- new best val perf: {:g} --> {:g} '
                                ''.format(best_val_perf, current_best))

                        best_val_perf = current_best
                        torch.save({'opt': args,
                                    'sd': model.state_dict(),
                                    'perf': best_val_perf, 'epoch': epoch,
                                    'opt_sd': optimizer.state_dict(),
                                    'tr_loss': tr_loss, 'step_num': step_num,
                                    'logging_loss': logging_loss},
                                args.model)
                    else:
                        early_stop_cnt += 1
                        log.log(f'EarlyStopping counter: {early_stop_cnt} out of {args.patience}')
                        if early_stop_cnt > args.patience:
                            early_stop_flag = True

            if early_stop_flag:
                log.log("EarlyStopping: Stop training")
                break
            epoch += 1
    
    log.log('best f1 {:g} in epoch {:g}'.format(current_best, curr_best_epoch))
    # wandb.finish()


if __name__ == '__main__':
    print('********* START *********')
    parser = argparse.ArgumentParser()

    # if not os.path.exists('ResNet'):
    #     os.makedirs('ResNet')
    parser.add_argument("--model",
                        default="/workplace/project/fire/ResNet/resnet50_80_5.pth")
    # parser.add_argument("--model",
    #                     default="/workplace/project/fire/ViT")                    

    parser.add_argument("--batch_num_pos", default=2)
    parser.add_argument("--batch_num_neg", default=2)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--sample", default=100)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--step_size", default=20)
    parser.add_argument("--scheduler_gamma", default=0.1)
    parser.add_argument("--test_epochs", default=1074)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--evaluate_threshold", default=0.55)
    parser.add_argument("--patience", default=50)

    args = parser.parse_args()

    # input=torch.randn(128,4,15,15).cuda()
    # output=model(input)#128*1
    # print('ok')
    train()