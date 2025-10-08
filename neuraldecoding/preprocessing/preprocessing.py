import numpy as np
import torch

import neuraldecoding.preprocessing.blocks

import time

class Preprocessing:
    def __init__(self, config):
        self.config = config
        self.pipeline = []

        order = self.config.get('order', [])
        content = self.config.get('content', {})
        
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
        current_data = data
        inter_pipeline_data = {}
        inter_pipeline_data.update(params)
        for step in self.pipeline:
            step_instance = step['instance']
            current_data, inter_pipeline_data = step_instance.transform(current_data, inter_pipeline_data)
        return current_data

    def preprocess_step(self, data, step_name, inter_pipeline_data = {'is_train': True}):
        for step in self.pipeline:
            if step['name'] == step_name:
                step_instance = step['instance']
                return step_instance.transform(data, inter_pipeline_data)
        
        raise ValueError(f"Step '{step_name}' not found in preprocessing pipeline")
    
class OnlinePreprocessing(Preprocessing):
    def __init__(self, config):
        super().__init__(config)
    
    def preprocess_pipeline(self, data, params = {'is_online': True}, publish_intermediate_steps = False, publish_func = None):
        current_data = data
        inter_pipeline_data = {}
        inter_pipeline_data.update(params)
        for step in self.pipeline:
            step_instance = step['instance']
            current_data, inter_pipeline_data = step_instance.transform_online(current_data, inter_pipeline_data)
            if publish_intermediate_steps:
                if publish_func is None:
                    raise ValueError("publish_func is required when publish_intermediate_steps is True")
                publish_func({'step': step['name'], 'data': current_data, 'interpipe': inter_pipeline_data}, is_intermediate_step = True, step_name=step["name"])
        if publish_func is None:
            raise ValueError("publish_func is required when publish_intermediate_steps is True")
        publish_func(current_data, is_intermediate_step=False)
        return current_data
