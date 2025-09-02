from omegaconf import DictConfig, ListConfig
from typing import Union

scheduler_struct = {
    'type': str,
    'is_iterative': bool,
    'params': DictConfig
}

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
        'content': DictConfig,
        'model_conf_append': ListConfig
}
