import network
import transform_data
import slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch import nn
import pandas as pd
import os
import argparse

def matrix_analysis(label_pred, label_true, cate_num):


    matrix = np.zeros([cate_num, cate_num])

    for i in range(cate_num):
        index = np.where(label_true == i)[0]
        label_term = label_pred[index]
        for j in range(cate_num):
            matrix[i, j] = len(np.where(label_term == j)[0])

    return matrix

def result_evaluation(matrix):

    label_num = matrix.shape[0]

    precision = [matrix[i, i] / sum(matrix[:, i]) for i in range(label_num)]
    recall = [matrix[i, i] / sum(matrix[i, :]) for i in range(label_num)]
    # f1_score = [2.0 / (1.0 / precision[i] + 1.0 / recall[i]) for i in range(label_num)]
    f1_score = [2.0 * precision[i] * recall[i] / (precision[i] + recall[i]) for i in range(label_num)]

    return precision, recall, f1_score

def parameter_setting(args):
    config = {}
    config['img_root'] = args.img_root
    config['spe_root'] = args.spe_root
    config['batch_size'] = args.batch_size
    config['data_file'] = args.data_file
    config['spe3D_max'] = args.spe3D_max
    config['img_feat_max'] = args.img_feat_max
    config['img_mean_std'] = {'img_mean': args.img_mean_std[0],
                             'img_std': args.img_mean_std[1]}
    config['catefile'] = args.catefile
    config['pretrained_model'] = args.pretrained_model
    config['cate_num'] = args.cate_num
    config['device'] = args.device


    return config


def get_dataloader(config):
    spe_transform = transforms.Compose([
        transform_data.Normalize_spe_xy(min_value=0, max_value=config['spe3D_max']),  # spe_3d max value
        transform_data.Numpy2Tensor()
    ])

    img_transform = transforms.Compose([
        transform_data.Normalize_img(mean=config['img_mean_std']['img_mean'], std=config['img_mean_std']['img_std']),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = slc_dataset.SLC_img_spe4D(txt_file=config['data_file'],
                                        img_dir=config['img_root'],
                                        spe_dir=config['spe_root'],
                                        catefile=config['catefile'],
                                        img_transform=img_transform,
                                        spe_transform=spe_transform)

    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=8)

    return dataloader


def joint_test(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('cuda:', os.environ["CUDA_VISIBLE_DEVICES"])

    cate_num = config['cate_num']
    pretrained_model = config['pretrained_model']
    net_joint = network.SLC_joint2(img_feat_max=config['img_feat_max'], num_classes=cate_num)
    net_joint.load_state_dict(torch.load(pretrained_model))

    net_joint.to(device)
    net_joint.eval()

    dataloader = get_dataloader(config)
    label2name = pd.read_csv(config['catefile'])

    acc_num = 0.0
    data_num = 0
    matrix = np.zeros([cate_num, cate_num])
    iter_val = iter(dataloader)
    for j in range(len(dataloader)):
        val_data = next(iter_val)
        val_img = val_data['img'].to(device)
        val_spe = val_data['spe'].to(device)
        val_label = val_data['label'].to(device)
        val_output = net_joint(val_spe, val_img)

        # val_loss = loss_func(val_output, val_label)
        _, pred = torch.Tensor.max(val_output, 1)
        acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
        data_num += val_label.size()[0]
        matrix += matrix_analysis(torch.squeeze(pred.cpu()).data.float(), val_label.cpu().data.float(), cate_num)

        # val_loss += loss_func(val_output, val_label).item()

    # val_loss /= len(dataloader)
    val_acc = acc_num / data_num
    [p, r, f] = result_evaluation(matrix)
    print(val_acc)
    for i in range(cate_num):
        print(label2name.loc[i]['catename'], '\t\t\t\t', '{:.4f}'.format(p[i]), '{:.4f}'.format(r[i]),
              '{:.4f}'.format(f[i]))
    print('f1-score-avg:', sum(f) / cate_num)
    print(matrix)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='arg_train_joint')

    parser.add_argument('--img_root', type=str, default='../data/slc_data/')
    parser.add_argument('--spe_root', type=str, default='../data/slc_spe4D_fft_12_spe3D/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_file', type=str, default='../data/slc_val_3.txt')
    parser.add_argument('--spe3D_max', type=float, default=0.18485471606254578)
    parser.add_argument('--img_feat_max', type=float, default=5.859713554382324)
    parser.add_argument('--img_mean_std', type=float, nargs='+', default=[0.29982, 0.07479776])
    parser.add_argument('--catefile', type=str, default='../data/slc_catename2label_cate8.txt')
    parser.add_argument('--pretrained_model', type=str, default='../model/slc_joint_deeper_3_F.pth')
    parser.add_argument('--cate_num', type=int, default=8)
    parser.add_argument('--device', default='0')

    args = parser.parse_args()
    config = parameter_setting(args)
    joint_test(config)