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
            'parameters': {
                'input_size': int,
                'num_outputs': int,
                'hidden_size': int,
                'num_layers': int,
                'rnn_type': str,
                'device': str,
                'hidden_noise_std': float,
                'dropout_input': bool,
                'drop_prob': float
            }
        },
        'optimizer': {
            'type': str,
            'params': {
                'lr': float,
                'weight_decay': float
            }
        },
        'scheduler': {
            'type': str,
            'params': {
                'step_size': int,
                'gamma': float
            }
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
        'data': {
            'data_path': str,
            'params': DictConfig
        }
    }