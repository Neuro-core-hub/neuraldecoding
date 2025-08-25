import os
import copy
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf
from hydra import initialize_config_dir, compose, initialize
from ..utils import verify_structure, parse_verify_config, compare_configs
import pandas as pd
from datetime import datetime
import hashlib
import time

class config:
    def __init__(self, config_path = None):
        '''
        only takes in absolute path to the config file or dir
        '''
        self.config = DictConfig({})
        self.original_config = DictConfig({})
        self.config_path = config_path
        self.update_history = pd.DataFrame(columns=['timestamp', 'entry', 'operation'])
        self.first_load = True
        if config_path:
            self.load(config_path)

    def __call__(self, section_name = None):
        if section_name is None:
            return self.config
        return self.parse_section(section_name, {})

    def load(self, config_path):
        '''
        only takes in absolute path to the config file or dir
        '''
        if not config_path.endswith('.yaml'):
            with initialize_config_dir(version_base=None, config_dir=config_path):
                cfg = compose("config")
        else:
            cfg = OmegaConf.load(config_path)
        
        self.config = cfg
        if self.config["hash_id"] is None:
            self.hash = hashlib.md5(datetime.now().strftime("%Y%m%d%H%M%S").encode('ascii')).hexdigest()
            self.config["hash_id"] = self.hash
        else:
            self.hash = self.config["hash_id"]
        if self.first_load:
            self.original_config = copy.deepcopy(cfg)
            self.first_load = False
        self.config_path = config_path
        
        # TODO: figure out what to do with hash if the config is updated, right now hash stays the same as the original one

    def update(self, key_path: str, value: Any, merge) -> None:
        '''
        Set a value in the configuration using dot notation. Saves update to history

        Args:
            key_path (str): Dot-separated path to the key (e.g., 'trainer.model.type').
            value (Any): Value to set.
            merge (bool): Whether to merge the value with existing data or replace it.
        '''
        OmegaConf.update(self.config, key_path, value, merge=merge)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history = pd.DataFrame({
            'timestamp': [current_time],
            'entry': [key_path],
            'operation': [{'value': value, 'merge': merge}]
        })
        self.update_history = pd.concat([self.update_history, history], ignore_index=True)

    def save(self, file_path = None, file_name = None):
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
            if not file_name.endswith('.yaml'):
                file_name += '.yaml'

            if self.config_path.endswith('.yaml'):
                file_path = os.path.dirname(self.config_path)
            else:
                file_path = self.config_path

        if file_path is not None and file_name is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            config_name = os.path.basename(self.config_path)
            file_name = f"{config_name}_{timestamp}.yaml"

        fpath = os.path.join(file_path, file_name)

        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with open(fpath, 'w') as f:
            OmegaConf.save(self.config, f)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history = pd.DataFrame({
            'timestamp': [current_time],
            'entry': [fpath],
            'operation': ["SAVE"]
        })
        self.update_history = pd.concat([self.update_history, history], ignore_index=True)

    def validate(self):
        # validate the structure of the whole config, not implemented yet due to configs are being changed frequently
        # TODO: impmlement this when the config structure is finalized
        raise NotImplementedError("config.validate is not implemented yet")

    def reset(self):
        self.config = copy.deepcopy(self.original_config)
        self.update_history = pd.DataFrame(columns=['timestamp', 'entry', 'operation'])

    def parse(self, section_name):
        return parse_verify_config(self.config, section_name)
    
    def has_changes(self, comparison_config = None):
        if comparison_config is None:
            return self.config != self.original_config
        else:
            if isinstance(comparison_config, config):
                comparison_config = comparison_config.config
            return self.config != comparison_config # apparently you can do this to dicts ^_^

    def get_history(self):
        return self.update_history

    def get_hash(self):
        return self.hash

    def get_changes(self, comparison_config = None):
        if comparison_config is None:
            comparison_config = self.original_config
        else:
            if isinstance(comparison_config, config):
                comparison_config = comparison_config.config
        return compare_configs(self.config, comparison_config)

    def get_readable(self):
        return OmegaConf.to_yaml(self.config)

    def get_value(self, key_path: str) -> Any:
        try:
            return OmegaConf.select(self.config, key_path)
        except:
            return KeyError(f"Key '{key_path}' not found in configuration")