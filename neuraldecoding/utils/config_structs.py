from omegaconf import DictConfig, ListConfig

decoder_struct = {
        'model': DictConfig
}
trainer_struct_nn = {
        'model': DictConfig,
        'optimizer': DictConfig,
        'scheduler': DictConfig,
        'loss_func': DictConfig,
        'training': DictConfig,
        'evaluation': DictConfig,
        'data': DictConfig
    }

trainer_struct_linear = {
        'model': DictConfig,
        'training': DictConfig,
        'evaluation': DictConfig,
        'data': DictConfig
    }
preprocessing_struct = {
        'order': ListConfig,
        'content': DictConfig
}
