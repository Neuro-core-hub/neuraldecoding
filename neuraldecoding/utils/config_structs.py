from omegaconf import DictConfig, ListConfig

decoder_struct = {'model': {'name': str, 
                                 'parameters': DictConfig, 
                                 'input_shape': ListConfig, 
                                 'output_shape': ListConfig}, 
                       'stabilization': {'name': str, 
                                         'parameters': DictConfig, 
                                         'date_0': str, 
                                         'date_k': str},
                       'fpath': str
                    }
trainer_struct = {
        'model': {
            'type': str,
            'parameters': DictConfig
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
            'print_every': int
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