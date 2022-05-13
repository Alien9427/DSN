import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import transform_data
from torch import optim, nn
import network
from slc_dataset import SLC_spe_4D
import numpy as np
import os.path
import argparse

"""
mapping_r4_r3.py
Map the 4-D signals to 3-D tensors, with the pre-trained cae model ../model/slc_spexy_cae_3.pth
The 3-D tensors are saved in ../data/spexy_data_3
"""

def parameter_setting(args):
    config = {}
    config['data_txt'] = args.data_txt
    config['save_dir'] = args.save_dir
    config['batchsize'] = args.batchsize
    config['pretrained_model'] = args.pretrained_model
    config['spe_dir'] = args.spe_dir
    config['catefile'] = args.catefile
    config['spe4D_min_max'] = {'spe4D_min': args.spe4D_min_max[0],
                               'spe4D_max': args.spe4D_min_max[1]}

    return config


def map3d(config):

    txt_file = config['data_txt']
    save_path = config['save_dir']
    batch_size = config['batchsize']

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    data_transforms = transforms.Compose([
        transform_data.Normalize_spe_xy(min_value=config['spe4D_min_max']['spe4D_min'], max_value=config['spe4D_min_max']['spe4D_max']),
        transform_data.Numpy2Tensor()
    ])
    dataset = SLC_spe_4D(txt_file=txt_file, spe_dir=config['spe_dir'], catefile=config['catefile'], spe_transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    net = network.SLC_spexy_CAE()
    net.load_state_dict(torch.load(config['pretrained_model']))

    spe_3D = np.zeros([batch_size, 128, 32, 32])

    for sample in dataloader:
        data = sample['spe']
        flag = 0
        for i, each_path in enumerate(sample['path']):
            cate = each_path.split('/')[0]
            if not os.path.exists(save_path + cate):
                os.mkdir(save_path + cate)
            if not os.path.exists(save_path + each_path):
                flag = 1

        if flag:
            for y in range(32):
                data_xy = data[:, :, y, :, :].reshape([data.shape[0] * 32, 1, 32, 32])
                data_out = net.get_encoder_features(data_xy).cpu().data.reshape([data.shape[0], 32, 128])
                spe_3D[:, :, :, y] = np.transpose(data_out, (0, 2, 1))

                for i, each_path in enumerate(sample['path']):
                    if not os.path.exists(save_path + each_path):
                        np.save(save_path + each_path, spe_3D[i, :, :, :])

                        print(each_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='arg_train')

    parser.add_argument('--data_txt', default='../data/mstar10_train_all.txt')
    parser.add_argument('--save_dir', default='../data/mstar_train_spe4D_fft_12_3D')
    parser.add_argument('--spe_dir', default='../data/mstar_train_spe4D_fft_12/')
    parser.add_argument('--pretrained_model', default='../model/mstar_cae_iter9000.pth')
    parser.add_argument('--catefile', default='../data/slc_catename2label_cate8.txt')
    parser.add_argument('--spe4D_min_max', type=float, nargs='+', default=[0.0011597341927439826, 10.628257178154184])
    parser.add_argument('--batchsize', type=int, default=2)


    args = parser.parse_args()
    config = parameter_setting(args)

    map3d(config)
