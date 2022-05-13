import network
import transform_data
import slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ImbalancedDatasetSampler
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from learning_schedule import param_setting_jointmodel2
from torch import optim
import argparse
import os

"""
train_joint.py
DSN training. Saved as ../model/slc_joint_deeper_3_F.pth
"""

def get_pretrained(img_model, net_joint):
    img_mapping = {'conv1':'pre_img.0',
                   'bn1':'pre_img.1',
                   'layer1':'pre_img.3',
                   'layer2':'pre_img.4'}

    for key in img_model.keys():
        if key[:5] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:5]] + key[5:]].data.copy_(img_model[key])
        elif key[:3] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:3]] + key[3:]].data.copy_(img_model[key])
        elif key[:6] in img_mapping.keys():
            net_joint.state_dict()[img_mapping[key[:6]] + key[6:]].data.copy_(img_model[key])
    return net_joint


def parameter_setting(args):
    config = {}
    config['img_root'] = {'train': args.img_root[0],
                            'val': args.img_root[1]}
    config['spe_root'] = {'train': args.spe_root[0],
                          'val': args.spe_root[1]}
    config['batch_size'] = {'train': args.batch_size[0],
                            'val': args.batch_size[1]}
    config['data_file'] = {'train': args.data_file[0],
                            'val': args.data_file[1]}
    config['spe3D_max'] = args.spe3D_max
    config['img_feat_max'] = args.img_feat_max
    config['img_mean_std'] = {'img_mean': args.img_mean_std[0],
                             'img_std': args.img_mean_std[1]}
    config['catefile'] = args.catefile
    config['save_model_path'] = args.save_model_path
    config['pretrained_model'] = args.pretrained_model
    config['img_model'] = args.img_model
    config['epoch_num'] = args.epoch_num
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

    dataset = {x: slc_dataset.SLC_img_spe4D(txt_file=config['data_file'][x],
                                            img_dir=config['img_root'][x],  # img
                                            spe_dir=config['spe_root'][x],  # frequency features
                                            catefile=config['catefile'],
                                            img_transform=img_transform,
                                            spe_transform=spe_transform)
               for x in ['train', 'val']}

    dataloaders = {}

    dataloaders['train'] = DataLoader(dataset['train'],
                                      batch_size=config['batch_size']['train'],
                                      sampler=ImbalancedDatasetSampler(dataset['train']),
                                      num_workers=8)
    dataloaders['val'] = DataLoader(dataset['val'],
                                    batch_size=config['batch_size']['val'],
                                    shuffle=True,
                                    num_workers=8)

    train_count_dict = {}
    for i in range(config['cate_num']):
        train_count_dict[i] = len(dataset['train'].data.loc[dataset['train'].data['label'] == i])

    loss_weight = [
        (1.0 - float(train_count_dict[i]) / float(sum(train_count_dict.values()))) * config['cate_num'] / (config['cate_num'] - 1)
        for i in range(config['cate_num'])]

    return dataloaders, loss_weight

def joint_train(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('cuda:', os.environ["CUDA_VISIBLE_DEVICES"])

    dataloader, loss_weight = get_dataloader(config)

    cate_num = config['cate_num']
    save_model_path = config['save_model_path']

    img_model = torch.load(config['img_model'])  # tsx pre-trained model for sar amplitude img. (gpu load)
    net_joint = network.SLC_joint2(img_feat_max=config['img_feat_max'], num_classes=cate_num)
    net_joint = get_pretrained(img_model, net_joint)

    if config['pretrained_model']:
        net_joint.load_state_dict(torch.load(config['pretrained_model']))

    net_joint.to(device)

    epoch_num = config['epoch_num']
    i = 0
    parameter_list = param_setting_jointmodel2(model=net_joint)
    optimizer = optim.SGD(parameter_list, lr=0.01, weight_decay=0.0005)

    # lr_list = [param_group['lr'] for param_group in optimizer.param_groups]  # pytorch 0.4.0
    loss_weight = torch.Tensor(loss_weight).to(device)  # pytorch 0.4.0
    loss_func = nn.CrossEntropyLoss(weight=loss_weight)

    writer = SummaryWriter('../log/' + save_model_path.split('/')[-1] + 'log')

    for epoch in range(epoch_num):
        for data in dataloader['train']:
            net_joint.train()
            optimizer.zero_grad()

            img_data = data['img'].to(device)
            spe_data = data['spe'].to(device)
            labels = data['label'].to(device)
            output = net_joint(spe_data, img_data)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()
            i += 1

        acc_num = 0.0
        data_num = 0
        val_loss = 0.0

        print('epoch ' + str(epoch + 1) + '\titer ' + str(i) + '\tloss ', loss.item())
        net_joint.eval()
        iter_val = iter(dataloader['val'])

        with torch.no_grad():
            for j in range(len(dataloader['val'])):
                val_data = next(iter_val)
                val_img = val_data['img'].to(device)
                val_spe = val_data['spe'].to(device)
                val_label = val_data['label'].to(device)
                val_output = net_joint(val_spe, val_img)

                # val_loss = loss_func(val_output, val_label)
                _, pred = torch.Tensor.max(val_output, 1)
                acc_num += torch.sum(torch.squeeze(pred) == val_label.data).float()
                data_num += val_label.size()[0]
                val_loss += loss_func(val_output, val_label).item()

        val_loss /= len(dataloader['val'])
        val_acc = acc_num / data_num
        print('test acc: ', val_acc.cpu().numpy())
        writer.add_scalars('loss', {'train': loss.item(),
                                    'val': val_loss},
                           epoch)
        writer.add_scalar('val_acc', val_acc, epoch)

        torch.save(net_joint.state_dict(), save_model_path + 'epoch' + str(epoch + 1) + '.pth')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_train_joint')

    parser.add_argument('--img_root', type=str, nargs='+', default=['../data/slc_data/', '../data/slc_data/'])
    parser.add_argument('--spe_root', type=str, nargs='+', default=['../data/slc_spe4D_fft_12_spe3D/', '../data/slc_spe4D_fft_12_spe3D/'])
    parser.add_argument('--batch_size', type=int, nargs='+', default=[64, 100])
    parser.add_argument('--data_file', type=str, nargs='+', default=['../data/slc_train_3.txt', '../data/slc_val_3.txt'])
    parser.add_argument('--spe3D_max', type=float, default=0.18485471606254578)
    parser.add_argument('--img_feat_max', type=float, default=5.859713554382324)
    parser.add_argument('--img_mean_std', type=float, nargs='+', default=[0.29982, 0.07479776])
    parser.add_argument('--catefile', type=str, default='../data/slc_catename2label_cate8.txt')
    parser.add_argument('--img_model', default='../model/tsx.pth')  # 训练好的模型存在这里
    parser.add_argument('--save_model_path', default='../model/slc_joint_')  # 训练好的模型存在这里
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--cate_num', type=int, default=8)
    parser.add_argument('--device', default='0')

    args = parser.parse_args()
    config = parameter_setting(args)
    joint_train(config)





