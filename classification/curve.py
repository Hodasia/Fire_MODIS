from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import torch
import argparse
import os
import numpy as np
from collections import Counter
import model_resnet as m_resnet
import logger
import utils

def plot_precision_recall_curve(model, args):
    model.eval()

    true_labels = []
    predictions = []

    index = 1
    num = args.test_epochs
    # thrd = args.evaluate_threshold

    with torch.no_grad():
        while index <= num:
            test_x, test_y = utils.sample_val_test(index, 'valid')

            ims = utils.nor2(test_x)
            input = torch.Tensor(ims).cuda()
            pred = model(input)
            pred = pred.detach().cpu().numpy()
            # pred[pred <= thrd] = 0
            # pred[pred > thrd] = 1

            true_labels.extend(test_y)
            predictions.extend(pred)

            index += 1
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    # print(precision, recall)
    np.seterr(divide='ignore',invalid='ignore')
    f1_scores = 2 * (precision * recall) / (precision + recall)
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print('optimal thrd according to p-r curve: {:f}'.format(optimal_threshold))

    # Plot the precision-recall curve
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.legend()
    plt.savefig("/workplace/project/fire/classification/thrd_img/resnet50_80_1.png")
    plt.show()

def sample_val_test(img_id, type):
    data = np.load(os.path.join('/workplace/dataset/MODIS_new/', type+'_80/'+str(img_id)+'.npy'))#9*2030*1354

    batch_x=data[:,:9]#12150*9*15*15
    batch_y=data[:,-1]#12150*1*15*15
    label = batch_y.copy()
    batch_y[batch_y<60]=0
    batch_y[batch_y>=60]=1
    batch_y=np.sum(batch_y,axis=(-1,-2))#12150*1
    batch_y[batch_y > 0]=1
    batch_y = batch_y.reshape(-1, 1)

    return batch_x,batch_y, label

def fpfn_samples(args, model, flag):
    model.eval()

    index=1
    num=args.test_epochs
    thrd=args.evaluate_threshold
    limit=1
    channel = 4
    label_final = []

    with torch.no_grad():
        while(index<=num):
            # print(index)
            test_x, test_y, label = sample_val_test(index, flag)

            ims = utils.nor2(test_x)  
            input = torch.Tensor(ims).cuda()
            pred = model(input)  # 8*9*101*101*1
            pred=pred.detach().cpu().numpy()
            pred_copy = pred.copy()
            # print(type(pred_copy))
            # print(pred_copy[962])
            pred[pred<=thrd]=0
            pred[pred>thrd]=1

            f=pred-test_y
            indices_fp = np.where(f == 1)
            indices_fn = np.where(f == -1)

            cnt=0
            print("*********** fp *************")
            for i in indices_fp[0]:
                cnt+=1
                print('###################')
                img_c = test_x[i, channel]
                idx = np.where(img_c == np.max(img_c))
                print('sub image {:d} in {:d}.npy, max in channel {:d}: {:g}'.format(i, index, channel, np.max(img_c)))
                print("prediction prob", repr(pred_copy[i][0]))
                print(label)
                # print("label", repr(label[i, 0, idx[0], idx[1]]))
                # label_final.extend(label[i, 0, idx[0], idx[1]])
                print(img_c)
                print(label[i, :])
                
                plt.imshow(img_c)
                plt.colorbar()
                plt.savefig("/workplace/project/fire/classification/fp_sample1.png")
                plt.show()

                if (cnt >= limit):
                    break

            # cnt=0
            # print("*********** fn *************")
            # for j in indices_fn[0]:
            #     print('###################')
            #     cnt+=1
            #     img_c = test_x[j, channel]
            #     idx = np.where(img_c == np.max(img_c))
            #     print('sub image {:d} in {:d}.npy, max in channel {:d}: {:g}'.format(j, index, channel, np.max(img_c)))
            #     print("prediction prob", repr(pred_copy[j][0]))
            #     print("label", repr(label[j, 0, idx[0], idx[1]]))
            #     print(img_c)
            #     print(label[j, :])
                
            #     plt.imshow(img_c)
            #     plt.colorbar()
            #     plt.show()

            #     if (cnt >= limit):
            #         break
            index += 1
    # print("labels:", label_final)
    # Count occurrences of each element
    # element_cnts = Counter(label_final)

    # Print the result
    # for element, count in element_cnts.items():
    #     print(f"Label {element} appears {count} times.")

def test():
    log = logger.Logger(args.model + '.log', on=True)
    log.log(str(args))
    log.log('***** p-r curve and optimal thrd *****')
    log.log('# test epoch: {:d}'.format(args.test_epochs))

    model = m_resnet.resnet50()
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict['sd'])
    model.cuda()

    # Plot precision-recall curve
    plot_precision_recall_curve(model, args)
    # fpfn_samples(args, model, 'valid')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # if not os.path.exists('ResNet'):
    #     os.makedirs('ResNet')
    parser.add_argument("--model",
                    default="/workplace/project/fire/ResNet/resnet50_80_1.pth")                   
    parser.add_argument("--test_epochs", default=1074)
    parser.add_argument("--evaluate_threshold", default=0.77)

    args = parser.parse_args()
    print('********** START **********')
    test()

# parser = argparse.ArgumentParser()

# parser.add_argument("--model",
#                 default="/workplace/project/fire/ResNet/resnet50.pth")                   
# parser.add_argument("--test_epochs", default=1074)
# parser.add_argument("--evaluate_threshold", default=0.7)

# args = parser.parse_known_args()[0]
# print('********** START **********')
# test(args)

