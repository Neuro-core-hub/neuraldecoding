from omegaconf import DictConfig, ListConfig
from typing import Union

scheduler_struct = {
    'type': str,
    'is_iterative': bool,
    'params': DictConfig
}

decoder_struct = {
        'model': {
            'type': str,
            'params': DictConfig
        },
        'fpath': str
}
trainer_struct_nn = {
        'model': {
            'type': str,
            'params': DictConfig
        },
        'optimizer': {
            'type': str,
            'params': DictConfig
        },
        'loss_func': {
            'type': str,
            'params': DictConfig
        },
        'training': {
            'num_epochs': Union[int, None],
            'max_iters': Union[int, None],
            'batch_size': int,
            'device': str,
            'print_results': bool,
            'print_every': int,
            'clear_cache': bool
        },
        'evaluation': {
            'metrics': ListConfig,
            'params': DictConfig
        },
        'data': {
            'data_path': str,
            'params': DictConfig
        }
    }

trainer_struct_linear = {
        'model': {
            'type': str,
            'params': DictConfig
        },
        'data': {
            'data_path': str,
            'params': DictConfig
        }
    }

preprocessing_struct = {
        'order': ListConfig,
        'content': DictConfig,
        'model_conf_append': ListConfig
}
