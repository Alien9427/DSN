def param_setting_jointmodel2(model):
    # fc_params = list(map(id, model.fc.parameters()))
    # post_params = list(map(id, model.post_slc))
    # pre_img_params = list(map(id, model.pre_img))
    # pre_spe_params = list(map(id, model.pre_spe))

    param_list = [{'params': model.fc.parameters(), 'lr': 1},
                  {'params': model.post_slc.parameters(), 'lr': 1},
                  {'params': model.pre_img.parameters(), 'lr': 0}
                  # {'params': model.pre_spe.parameters(), 'lr': 0.1}
                  ]

    return param_list
