import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import transform_data
import network
from slc_dataset import SLC_cae_spe4D
from torch import optim, nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import argparse
import os
import time


"""
train_cae.py
Train the stacked convolutional auto-encoder model. Saved as ../model/slc_spexy_cae_3.pth
"""


def parameter_setting(args):
    config = {}
    config['data_root'] = {'train': args.data_root[0],
                            'val': args.data_root[1]}
    config['batch_size'] = {'train': args.batch_size[0],
                            'val': args.batch_size[1]}
    config['data_file'] = {'train': args.data_file[0],
                            'val': args.data_file[1]}
    config['spe4D_min_max'] = {'spe4D_min': args.spe4D_min_max[0],
                               'spe4D_max': args.spe4D_min_max[1]}
    config['catename2label'] = args.catename2label
    config['save_model_path'] = args.save_model_path
    config['pretrained_model'] = args.pretrained_model
    config['epoch_num'] = args.epoch_num
    config['device'] = args.device


    return config

def get_dataloader(config):

    data_transforms = transforms.Compose([
        transform_data.Normalize_spe_xy(min_value=config['spe4D_min_max']['spe4D_min'], max_value=config['spe4D_min_max']['spe4D_max']),
        transform_data.Numpy2Tensor_img(1)
    ])
    dataset = {x: SLC_cae_spe4D(txt_file=config['data_file'][x], spe_dir=config['data_root'][x],
                                catefile=config['catename2label'],
                                spe_transform=data_transforms)
               for x in ['train', 'val']}
    dataloader = {x: DataLoader(dataset[x],
                                batch_size=config['batch_size'][x],
                                shuffle=True,
                                num_workers=12)
                  for x in ['train', 'val']}

    return dataloader

def cae_train(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('cuda:', os.environ["CUDA_VISIBLE_DEVICES"])

    dataloader = get_dataloader(config)
    net = network.SLC_spexy_CAE()

    if config['pretrained_model']:
        net.load_state_dict(torch.load(config['pretrained_model']), strict=False)
    net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.00001, weight_decay=0.0005)
    loss_func = nn.MSELoss()

    params = list(net.parameters())
    writer = SummaryWriter('../log/' + config['save_model_path'].split('/')[-1] + 'log')

    i = 0
    iter_val = iter(dataloader['val'])

    print('initialization...')

    for epoch in range(config['epoch_num']):
        print('training epoch... ')
        for sample in dataloader['train']:
            data = sample['spe']
            # print('load data ...')

            optimizer.zero_grad()
            output = net(data.to(device))
            loss = loss_func(output, data.to(device))
            loss.backward()
            optimizer.step()
            # print(params[0].grad)

            if i % 10 == 0:
                print('epoch ' + str(epoch + 1) + '\titer ' + str(i) + '\tloss ', loss.item())
                net.eval()
                try:
                    val_sample = next(iter_val)
                except StopIteration:
                    iter_val = iter(dataloader['val'])
                    val_sample = next(iter_val)

                val_data = val_sample['spe']

                val_output = net(val_data.to(device))
                val_loss = loss_func(val_output, val_data.to(device))

                writer.add_scalars('loss', {'train': loss.item(),
                                            'val': val_loss.item()},
                                   i)

                # print(params[0].grad[0,0,:,:], params[3].grad[0,0,:,:], params[-1].grad)
                if i % 100 == 0:
                    # for j in range(batch_size['val']):
                    #     plt.subplot(2, 5, j + 1)
                    #     plt.imshow(val_data[j].reshape([32, 32]), cmap=plt.cm.jet)
                    #
                    #     plt.subplot(2, 5, j + 6)
                    #     plt.imshow(val_output.cpu().detach()[j].reshape([32, 32]), cmap=plt.cm.jet)
                    val_imgs = torch.cat((val_data, val_output.cpu()))
                    val_imgs = make_grid(val_imgs, nrow=10, scale_each=True, pad_value=10)
                    writer.add_image('imgs', val_imgs, i)

                    if i % 1000 == 0:
                        torch.save(net.state_dict(), config['save_model_path'] + 'iter' + str(i) + '.pth')
            i += 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_train_cae')

    parser.add_argument('--batch_size', type=int, nargs='+', default=[1000, 10])
    parser.add_argument('--spe4D_min_max', type=float, nargs='+', default=[0.0011597341927439826, 10.628257178154184])
    parser.add_argument('--data_file', type=str, nargs='+', default=['../data/slc_cae_train_3.txt', '../data/slc_cae_val_3.txt'])
    parser.add_argument('--data_root', type=str, nargs='+', default=['../data/slc_spe4D_fft_12/', '../data/slc_spe4D_fft_12/'])
    parser.add_argument('--catename2label', type=str, default='../data/slc_catename2label_cate8.txt')
    parser.add_argument('--save_model_path', default='../model/slc_cae_')  # 训练好的模型存在这里
    parser.add_argument('--pretrained_model', default='../model/slc_spexy_cae_3.pth')
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--device', default='0')

    args = parser.parse_args()
    config = parameter_setting(args)

    cae_train(config)

