from omegaconf import DictConfig, ListConfig

decoder_struct = {
        'model': {
            'type': str,
            'params': DictConfig
        },
        'fpath': str
}
trainer_struct = {
        'model': {
            'type': str,
            'params': DictConfig
        },
        'optimizer': {
            'type': str,
            'params': DictConfig
        },
        'scheduler': {
            'type': str,
            'params': DictConfig
        },
        'loss_func': {
            'type': str,
            'params': DictConfig
        },
        'training': {
            'num_epochs': int,
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
preprocessing_struct = {
        'order': ListConfig,
        'content': DictConfig
}
