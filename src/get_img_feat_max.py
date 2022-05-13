import network
import transform_data
import slc_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sampler import ImbalancedDatasetSampler
import numpy as np
import torch
import argparse
import os

def parameter_setting(args):
    config = {}
    config['img_root'] = args.img_root
    config['batch_size'] = args.batch_size
    config['data_file'] = args.data_file
    config['img_mean_std'] = {'img_mean': args.img_mean_std[0],
                             'img_std': args.img_mean_std[1]}
    config['catefile'] = args.catefile
    config['img_model'] = args.img_model
    config['cate_num'] = args.cate_num
    config['device'] = args.device


    return config

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


def get_img_feature(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('cuda:', os.environ["CUDA_VISIBLE_DEVICES"])

    txt_file = config['data_file']
    batch_size = config['batch_size']
    cate_num = config['cate_num']

    img_transform = transforms.Compose([
        transform_data.Normalize_img(mean=config['img_mean_std']['img_mean'], std=config['img_mean_std']['img_std']),
        transform_data.Numpy2Tensor_img(3)
    ])

    dataset = slc_dataset.SLC_img(txt_file=txt_file,
                                  root_dir=config['img_root'],
                                  catefile=config['catefile'],
                                  transform=img_transform)

    dataloaders = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=8,
                             drop_last=False)

    img_model = torch.load(config['img_model'])
    net_joint = network.SLC_joint2_img(cate_num)
    net_joint = get_pretrained(img_model, net_joint)

    net_joint.to(device)

    np_img_feature = np.zeros([0, 128, 16, 16])

    with torch.no_grad():
        for data in dataloaders:
            img_data = data['data'].to(device)
            img_features = net_joint.pre_img_features(img_data)
            np_img_feature = np.concatenate((np_img_feature, img_features.cpu().data.numpy()), axis=0)
    print('img_feat_max: ', np.max(np_img_feature))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='arg_train')

    parser.add_argument('--img_root', type=str, default='../data/slc_data/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--data_file', type=str, default='../data/slc_train_3.txt')
    parser.add_argument('--img_mean_std', type=float, nargs='+', default=[0.29982, 0.07479776])

    parser.add_argument('--catefile', type=str, default='../data/slc_catename2label_cate8.txt')
    parser.add_argument('--img_model', default='../model/tsx.pth')

    parser.add_argument('--cate_num', type=int, default=8)
    parser.add_argument('--device', default='1')

    args = parser.parse_args()
    config = parameter_setting(args)
    get_img_feature(config)





