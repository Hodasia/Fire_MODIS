import torch
import argparse
import os
import model_resnet as m_resnet
import logger
import utils
import wandb

def test():
    log = logger.Logger(args.model + '.log', on=True)
    log.log(str(args))
    log.log('***** test *****')
    log.log('# test epoch: {:d}'.format(args.test_epochs))
    log.log('# threshold: {:g}'.format(args.evaluate_threshold))

    model = m_resnet.resnet50()
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict['sd'])
    model.cuda()

    tp, fp, fn=utils.eval(args, model, 'test')
    log.log('test tp2 {:3d} | fp2 {:3d} | fn2 {:3d}'.format(tp, fp, fn))
    if (tp + fp > 0 and tp + fn > 0):
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if (p + r > 0):
            f1 = 2 * p * r / (p + r)
            fa=1-p
            log.log('test f1 {:8.4f}| test precision {:8.4f}| test recall {:8.4f} | test fa {:8.4f}'.format(f1, p, r, fa))

# def eval(args, model, type):
#     model.eval()
#     tp=0
#     fp=0
#     fn=0

#     index=1
#     num=args.test_epochs
#     thrd=args.evaluate_threshold
#     with torch.no_grad():
#         while(index<=num):
#             # print(index)
#             test_x, test_y = utils.sample_val_test(index, type)

#             ims = utils.nor2(test_x)  
#             input = torch.Tensor(ims).cuda()
#             pred = model(input)  # 8*9*101*101*1
#             pred=pred.detach().cpu().numpy()
#             pred[pred<=thrd]=0
#             pred[pred>thrd]=1

#             tpp,fpp,fnn= utils.cal_hit(pred,test_y)
#             tp+=tpp
#             fp+=fpp
#             fn+=fnn

#             index += 1

#     return tp,fp,fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    if not os.path.exists('ResNet'):
        os.makedirs('ResNet')
    parser.add_argument("--model",
                        default="/workplace/project/fire/ResNet/resnet50_80_3.pth")                   
    parser.add_argument("--test_epochs", default=1074)
    parser.add_argument("--evaluate_threshold", default=0.7)

    args = parser.parse_args()
    test()