import numpy as np
import os
from slc_functions import gen_spectrogram_2
import pandas as pd
import random
import glob
import argparse


def gen_all_spec(slc_root, spe_root, win):
    slc_list = os.listdir(slc_root)
    if not os.path.exists(spe_root):
        os.mkdir(spe_root)
    for cate in slc_list:
        if not os.path.exists(spe_root + cate):
            os.mkdir(spe_root + cate)

        data_list = os.listdir(slc_root + cate)
        for data in data_list:
            if not os.path.exists(spe_root + cate + '/' + data) and data[-3:] == 'npy':
                print(data)
                slc_data = np.load(slc_root + cate + '/' + data)
                spectrogram = np.log(1+np.abs(gen_spectrogram_2(slc_data, win)))

                np.save(spe_root + cate + '/' + data, spectrogram)


def get_range_spec(spe_root):
    max_value = 0
    min_value = 100
    spe_list = os.listdir(spe_root)
    for cate in spe_list:
        spec_list = os.listdir(spe_root + cate)
        for spec in spec_list:
            spectrogram = np.load(spe_root + cate + '/' + spec)
            max_value = max(max_value, spectrogram.max())
            min_value = min(min_value, spectrogram.min())

    print('spe4D_min_max: ', min_value, max_value)

def get_spe3D_max(spe_root):
    max_value = 0
    spe_list = os.listdir(spe_root)
    for cate in spe_list:
        spec_list = os.listdir(spe_root + cate)
        for spec in spec_list:
            spectrogram = np.load(spe_root + cate + '/' + spec)
            max_value = max(max_value, spectrogram.max())

    print('spe3D_max: ', max_value)

def gen_train_val(data_root):
    val_ratio = 0.7

    df_train = pd.DataFrame(columns=['path', 'catename'])
    df_val = pd.DataFrame(columns=['path', 'catename'])

    # val_num = 95
    for cate in os.listdir(data_root):
        data_list = os.listdir(data_root + cate)
        random.shuffle(data_list)
        val_num = int(len(data_list) * val_ratio)
        for i, item in enumerate(data_list):
            if i < val_num:
                df_val.loc[len(df_val) + 1] = [cate + '/' + item, cate]
            else:
                df_train.loc[len(df_train) + 1] = [cate + '/' + item, cate]

    df_train.to_csv('../data/slc_train_22.txt', index=False)
    df_val.to_csv('../data/slc_val_22.txt', index=False)

def get_mean_std_img(slc_root):
    img_list = os.listdir(slc_root)
    mean = 0
    std = 0
    count = 0
    for cate in img_list:
        slc_list = glob.glob(slc_root + cate + '/*.npy')
        for slc in slc_list:
            slc_data = np.load(slc)
            data = np.log2(np.abs(slc_data) + 1) / 16 # slc data
            # data = np.log(1+np.abs(slc_data)) # mstar
            mean += data.mean()
            std += data.std()
            count += 1

    print('img_mean_std:', mean / count, std / count)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='arg_train_joint')

    parser.add_argument('--slc_root', type=str, default=None)
    parser.add_argument('--spe4D_root', type=str, default=None)
    parser.add_argument('--spe3D_root', type=str, default=None)
    parser.add_argument('--win', type=float, default=0.5)

    args = parser.parse_args()

    slc_root = args.slc_root
    spe4D_root = args.spe4D_root
    spe3D_root = args.spe3D_root
    win = args.win

    if slc_root:
        print('get img_mean_std...')
        get_mean_std_img(slc_root)  # get img_mean_std
        if spe4D_root:
            print('generate 4D signals...')
            gen_all_spec(slc_root, spe4D_root, win) # generate the 4-D signals
            print('get spe4D_min_max...')
            get_range_spec(spe4D_root) # get spe4D_min_max

    if spe3D_root:
        print('get spe3D_max...')
        get_spe3D_max(spe3D_root) # get spe3D_max
