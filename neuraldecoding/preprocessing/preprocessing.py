import numpy as np
import torch

import neuraldecoding.preprocessing.blocks

import time

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
            
            preprocessing_class = getattr(neuraldecoding.preprocessing.blocks, step_type)
            preprocessing_instance = preprocessing_class(**step_params)
            
            self.pipeline.append({
                'name': step_name,
                'type': step_type,
                'instance': preprocessing_instance
            })
    
    def preprocess_pipeline(self, data, params = {'is_train': True}):
        current_data = data.copy()
        inter_pipeline_data = {}
        inter_pipeline_data.update(params)
        for step in self.pipeline:
            step_instance = step['instance']
            current_data, inter_pipeline_data = step_instance.transform(current_data, inter_pipeline_data)
        return current_data

    def preprocess_step(self, data, step_name, inter_pipeline_data = {}, params = {'is_train': True}):
        for step in self.pipeline:
            if step['name'] == step_name:
                step_instance = step['instance']
                return step_instance.transform(data, inter_pipeline_data, **params)
        
        raise ValueError(f"Step '{step_name}' not found in preprocessing pipeline")