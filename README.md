# Deep SAR-Net: Learning objects from signals

`src/script.sh`:

Step 1: generate the 4D signal `spe4D` via fft based time-frequency analysis, output `spe4D_min_max` values and  `img_mean_std` values

```
python data_process.py --slc_root ../data/slc_data/ \                 # single look complex data dir
                       --spe4D_root ../data/slc_spe4D_fft_12/ \       # 4D TF signal dir
                       --win 0.5                                      # hamming window size (propotion of slc_img, 0.5 or 0.25)
```

Step 2: train cae model

```
python train_cae.py --data_file ../data/slc_cae_train_3.txt ../data/slc_cae_val_3.txt \
                    --data_root ../data/slc_spe4D_fft_12/ ../data/slc_spe4D_fft_12/ \
                    --catename2label ../data/slc_catename2label_cate8.txt \
                    --save_model_path ../model/slc_cae_12_ \
                    --pretrained_model ../model/slc_spexy_cae_3.pth \
                    --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
                    --device 0
```

Step 3: generate spatially aligned frequency features `spe3D` using cae model

```
python mapping_r4_r3.py --data_txt ../data/slc_cate8_all.txt \
                        --save_dir ../data/slc_spe4D_fft_12_spe3D/ \            # spe3D features
                        --spe_dir ../data/slc_spe4D_fft_12/ \
                        --pretrained_model ../model/slc_spexy_cae_3.pth \
                        --catefile ../data/slc_catename2label_cate8.txt \
                        --spe4D_min_max 0.0011597341927439826 10.628257178154184 \
                        --batchsize 2
```

Step 4: get `spe3D_max` and `img_feat_max` for feature normalization

```
python data_process.py --spe3D_root ../data/slc_spe4D_fft_12_spe3D/

python get_img_feat_max.py --img_root ../data/slc_data/ \
                           --data_file ../data/slc_train_3.txt \
                           --img_mean_std 0.29982 0.07479776 \
                           --catefile ../data/slc_catename2label_cate8.txt \
                           --cate_num 8 \
                           --device 0
```

Step 5: train deep network 3

```
python train_joint.py --img_root ../data/slc_data/ ../data/slc_data/ \
                      --spe_root ../data/slc_spe4D_fft_12_spe3D/ ../data/slc_spe4D_fft_12_spe3D/ \
                      --data_file ../data/slc_train_3.txt ../data/slc_val_3.txt \
                      --spe3D_max 0.18485471606254578 \
                      --img_feat_max 5.859713554382324 \
                      --img_mean_std 0.29982 0.07479776 \
                      --catefile ../data/slc_catename2label_cate8.txt \
                      --img_model ../model/tsx.pth \
                      --save_model_path ../model/slc_joint_ \
                      --epoch_num 100 \
                      --cate_num 8 \
                      --device 0
```

<img width="736" alt="image" src="https://user-images.githubusercontent.com/8330403/168397893-f7bdef26-5b77-447c-92df-dc157a7a6a4f.png">


```
@article{dsn2020,
title = {Deep SAR-Net: Learning objects from signals},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {161},
pages = {179-193},
year = {2020},
issn = {0924-2716},
author = {Z. Huang and M. Datcu and Z. Pan and B. Lei},
}
```
