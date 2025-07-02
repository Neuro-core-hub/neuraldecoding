from omegaconf import DictConfig, ListConfig

decoder_struct = {
        'model': {
            'type': str,
            'params': DictConfig
        },
        'stabilization': {
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
        'stabilization': {
            'type': str, 
            'params': DictConfig
        },
        'data': {
            'data_path': str,
            'params': DictConfig
        }
    }
preprocessing_struct = {
    'preprocessing_trainer':{
        'order': ListConfig,
        'content': DictConfig},
    'preprocessing_decoder': {
        'order': ListConfig,
        'content': DictConfig}
}
