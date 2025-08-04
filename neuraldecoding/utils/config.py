import os
import copy
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf
from neuraldecoding.utils.config_structs import (
    decoder_struct,
    trainer_struct_nn,
    trainer_struct_linear,
    preprocessing_struct
)
from hydra import initialize, compose
from neuraldecoding.utils.config_parser import verify_structure, parse_verify_config, compare_configs
import pandas as pd
from datetime import datetime

class config:
    def __init__(self, config_path = None):
        self.config = DictConfig({})
        self.original_config = DictConfig({})
        self.config_path = config_path
        self.update_history = pd.DataFrame(columns=['entry', 'operation'])
        self.first_load = True
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        if os.path.isdir(config_path):
            with initialize(version_base=None, config_path=config_path):
                cfg = compose("config")
        else:
            cfg = OmegaConf.load(config_path)
        
        self.config = cfg
        if self.first_load:
            self.original_config = copy.deepcopy(cfg)
            self.first_load = False
        self.config_path = config_path

    def __call__(self, section_name = None):
        if section_name is None:
            return self.config
        return self.parse_section(section_name, {})
    
    def validate_config(self):
        raise NotImplementedError("Not implemented yet")

    def parse_section(self, section_name):
        if section_name not in self.config:
            raise KeyError(f"Section '{section_name}' not found in configuration")
        return parse_verify_config(self.config, section_name)
    
    def get_value(self, key_path: str) -> Any:
        try:
            return OmegaConf.select(self.config, key_path)
        except:
            return KeyError(f"Key '{key_path}' not found in configuration")

    def update_value(self, key_path: str, value: Any, merge) -> None:
        '''
        Set a value in the configuration using dot notation. Saves update to history

        Args:
            key_path (str): Dot-separated path to the key (e.g., 'trainer.model.type').
            value (Any): Value to set.
            merge (bool): Whether to merge the value with existing data or replace it.
        '''
        OmegaConf.update(self.config, key_path, value, merge=merge)
        history = pd.DataFrame({
            'entry': [key_path],
            'operation': [{'value': value, 'merge': merge}]
        })
        self.update_history = pd.concat([self.update_history, history], ignore_index=True)

    def get_history(self):
        return self.update_history
    
    def save_config(self, file_path = None, file_name = None):
        '''
        if file path not given, save to config directory with timestamp
        '''
        if file_path is None and file_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            config_dir = self.config_path
            config_name = os.path.basename(self.config_path)
            file_path = config_dir
            file_name = f"{config_name}_{timestamp}.yaml"
            
        if file_path is None and file_name is not None:
            file_path = os.path.dirname(self.config_path)

        if file_path is not None and file_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            config_name = os.path.basename(self.config_path)
            file_name = f"{config_name}_{timestamp}.yaml"

        fpath = os.path.join(file_path, file_name)

        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        # Save configuration
        with open(fpath, 'w') as f:
            OmegaConf.save(self.config, f)
        history = pd.DataFrame({
            'entry': [fpath],
            'operation': ["SAVE"]
        })
        self.update_history = pd.concat([self.update_history, history], ignore_index=True)

    def reset_to_original(self):
        self.config = copy.deepcopy(self.original_config)
    
    def has_changes(self):
        return self.config != self.original_config
    
    def get_changes(self):
        return compare_configs(self.config, self.original_config)
