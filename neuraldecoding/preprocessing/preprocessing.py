import numpy as np
import torch

import neuraldecoding.preprocessing.wrapper


method_reg = {
    "normalization": neuraldecoding.utils.data_tools.normalize,
    "add_history": neuraldecoding.utils.data_tools.add_history,
    "stabilization": neuraldecoding.utils.data_tools.stabilization,
}

class Preprocessing:
    def __init__(self, config):
        self.config = config
        self.pipeline = []

        order = self.config['order']
        content = self.config['content']
        
        for step_name in order:
            if step_name not in content:
                raise ValueError(f"Step '{step_name}' not found in config content")
            
            step_config = content[step_name]
            step_type = step_config['type']
            step_params = step_config.get('params', {})
            
            if step_type not in method_reg: #To be changed to wrapper
                raise ValueError(f"Unknown preprocessing type: {step_type}")
            
            preprocessing_class = method_reg[step_type] #To be changed to wrapper
            preprocessing_instance = preprocessing_class(**step_params)
            
            self.pipeline.append({
                'name': step_name,
                'type': step_type,
                'instance': preprocessing_instance
            })
    
    def preprocess(self, data, params = {'is_train': True}):
        current_data = data.copy()
        
        for step in self.pipeline_steps:
            step_instance = step['instance']
            current_data = step_instance.transform(current_data, **params)
        
        return current_data
