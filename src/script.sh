#!/usr/bin/env bash

#python data_process.py --slc_root ../data/slc_data/ \
#                       --spe4D_root ../data/slc_spe4D_fft_12/ \
#                       --win 0.5

#python train_cae.py --data_file ../data/slc_cae_train_3.txt ../data/slc_cae_val_3.txt \
#                    --data_root ../data/slc_spe4D_fft_12/ ../data/slc_spe4D_fft_12/ \
#                    --catename2label ../data/slc_catename2label_cate8.txt \
#                    --save_model_path ../model/slc_cae_12_ \
#                    --pretrained_model ../model/slc_spexy_cae_3.pth \
#                    --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
#                    --device 0

# python mapping_r4_r3.py --data_txt ../data/slc_cate8_all.txt \
#                         --save_dir ../data/slc_spe4D_fft_12_spe3D/ \
#                         --spe_dir ../data/slc_spe4D_fft_12/ \
#                         --pretrained_model ../model/slc_spexy_cae_3.pth \
#                         --catefile ../data/slc_catename2label_cate8.txt \
#                         --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
#                         --batchsize 2

# python data_process.py --spe3D_root ../data/slc_spe4D_fft_12_spe3D/

#python get_img_feat_max.py --img_root ../data/slc_data/ \
#                           --data_file ../data/slc_train_3.txt \
#                           --img_mean_std 0.29982 0.07479776 \
#                           --catefile ../data/slc_catename2label_cate8.txt \
#                           --cate_num 8 \
#                           --device 0

# python train_joint.py --img_root ../data/slc_data/ ../data/slc_data/ \
#                       --spe_root ../data/slc_spe4D_fft_14_spe3D/ ../data/slc_spe4D_fft_14_spe3D/ \
#                       --data_file ../data/slc_train_3.txt ../data/slc_val_3.txt \
#                       --spe3D_max 0.18485471606254578 \
#                       --img_feat_max 5.859713554382324 \
#                       --img_mean_std 0.29982 0.07479776 \
#                       --catefile ../data/slc_catename2label_cate8.txt \
#                       --img_model ../model/tsx.pth \
#                       --save_model_path ../model/slc_joint_ \
#                       --epoch_num 100 \
#                       --cate_num 8 \
#                       --device 0

#python ttest_joint.py --img_root ../data/slc_data/ \
#                      --spe_root ../data/slc_spe4D_fft_12_spe3D/ \
#                      --data_file ../data/slc_val_3.txt \
#                      --spe3D_max 0.18485471606254578 \
#                      --img_feat_max 5.859713554382324 \
#                      --img_mean_std 0.29982 0.07479776 \
#                      --catefile ../data/slc_catename2label_cate8.txt \
#                      --pretrained_model ../model/slc_joint_deeper_3_F.pth \
#                      --cate_num 8 \
#                      --device 0