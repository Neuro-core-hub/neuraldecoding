import copy

import numpy as np
import torch
from omegaconf import OmegaConf

from neuraldecoding.preprocessing import blocks

import time

class Preprocessing:
    def __init__(self, config):
        self.config = config
        self.pipeline = []
        self.saved_data = {}

        order = self.config.get('order', [])
        content = self.config.get('content', {})
        
        for step_name in order:
            if step_name not in content:
                raise ValueError(f"Step '{step_name}' not found in config content")
            
            step_config = content[step_name]
            step_type = step_config['type']
            step_params = step_config.get('params', {})
            
            preprocessing_class = getattr(blocks, step_type)
            preprocessing_instance = preprocessing_class(**step_params)
            
            self.pipeline.append({
                'name': step_name,
                'type': step_type,
                'instance': preprocessing_instance
            })
    
    def preprocess_pipeline(self, data, params = {'is_train': True}):
        current_data = data
        inter_pipeline_data = {'save_keys_ram': []}
        inter_pipeline_data.update(params)
        for step in self.pipeline:
            step_instance = step['instance']
            current_data, inter_pipeline_data = step_instance.transform(current_data, inter_pipeline_data)
        for key in inter_pipeline_data.get('save_keys_ram', []):
            self.saved_data[key] = inter_pipeline_data[key]
        self.interpipeline_data = inter_pipeline_data
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


def merge_online_neural_selection(online_cfgs, location_neural=None):
    """
    Share one online pipeline between per-DoF decoders (dofDecoder) that read
    different neural channels, e.g. index on [2, 6] and wrist on [16..30].

    The online preprocessing node publishes one feature vector for all DoFs. This
    returns a pipeline that selects the union of every DoF's channels, plus, per
    DoF, the columns of that vector holding its own channels in training order.
    Valid because the pipelines may differ only in the channel list, and the
    features (MAV etc.) are computed per channel.

    Args:
        online_cfgs: one `preprocessing_online` config per DoF, in DoF order.
        location_neural: data key the channel selector acts on (default: the
            first config's `location_neural`, else "neural").

    Returns:
        (merged_cfg, columns): columns[i] is None when every DoF already selects
        the same channels, so single-channel-set setups behave exactly as before.
    """
    plain = [OmegaConf.to_container(c, resolve=True) if OmegaConf.is_config(c) else copy.deepcopy(c)
             for c in online_cfgs]
    if location_neural is None:
        location_neural = plain[0].get("location_neural", "neural")

    selections, skeletons = [], []
    for i, cfg in enumerate(plain):
        steps = copy.deepcopy([cfg["content"][name] for name in cfg["order"]])
        selectors = []
        for step in steps:
            params = step.get("params") or {}
            locs = params.get("location") or []
            locs = [locs] if isinstance(locs, str) else list(locs)
            if step["type"] == "IndexSelectorBlock" and location_neural in locs:
                selectors.append(params)
        if len(selectors) > 1:
            raise ValueError(f"DoF {i}: expected at most one IndexSelectorBlock on '{location_neural}' "
                             f"in preprocessing_online, found {len(selectors)}")
        selections.append(list(selectors[0]["indices"]) if selectors else None)
        if selectors:
            selectors[0]["indices"] = None
        for step in steps:
            # `channels` is documentation only; the extractor sizes itself from the data.
            (step.get("params") or {}).get("feature_extractor_config", {}).pop("channels", None)
        skeletons.append((cfg.get("bin_size_ms"), cfg.get("location_neural"), steps))

    if all(sel == selections[0] for sel in selections):
        return OmegaConf.create(plain[0]), [None] * len(plain)
    if any(sel is None for sel in selections):
        raise ValueError("DoFs select different neural channels but at least one has no channel "
                         "selector, so their channels cannot be merged")
    if any(sk != skeletons[0] for sk in skeletons[1:]):
        raise ValueError("Per-DoF preprocessing_online pipelines differ in more than their neural "
                         "channel selection, so one preprocessing node cannot serve them all")

    union = sorted(set().union(*selections))
    merged = plain[0]
    for name in merged["order"]:
        step = merged["content"][name]
        params = step.get("params") or {}
        locs = params.get("location") or []
        locs = [locs] if isinstance(locs, str) else list(locs)
        if step["type"] == "IndexSelectorBlock" and location_neural in locs:
            params["indices"] = union
        fe_cfg = params.get("feature_extractor_config")
        if fe_cfg is not None and "channels" in fe_cfg:
            fe_cfg["channels"] = len(union)
    columns = [[union.index(ch) for ch in sel] for sel in selections]
    return OmegaConf.create(merged), columns
