from omegaconf import DictConfig, ListConfig
from neuraldecoding.utils.config_structs import decoder_struct, trainer_struct_nn, trainer_struct_linear, preprocessing_struct

def verify_structure(content: DictConfig, struct: dict) -> tuple[bool, str]:
    """
    Recursively verify if the content dictionary matches the structure and types defined in struct.

    Parameters:
    content (DictConfig): The content to verify.
    struct (dict): The structure definition where keys are expected keys and values are types or nested structures.

    Returns:
    (bool, str):bool: True if the content matches the structure, False otherwise.
                str: An error message if the content does not match the structure, None if it matches.
    """
    for key, value_type in struct.items():
        if key not in content:
            return False, f"Missing key: {key}"
        if isinstance(value_type, DictConfig) or isinstance(value_type, dict):
            if not isinstance(content[key], DictConfig):
                return False, f"Key '{key}' is not a dictionary as expected."
            is_valid, error = verify_structure(content[key], value_type)
            if not is_valid:
                return False, error
        else:
            if not isinstance(content[key], value_type):
                return False, f"Key '{key}' has incorrect type. Expected {value_type}, got {type(content[key])}."
    return True, None

def parse_verify_config(config: DictConfig, section_name: str) -> dict:
    """
    Take in a config dict and return the content of a specific section.

    Parameters:
    config (DictConfig): Config dictionary
    section_name (str): The section name to extract from the config file.

    Returns:
    dict: The content of the specified section in the config file.
    """
    content = config

    if section_name not in content:
        raise ValueError(f"Section '{section_name}' not found in the config file.")
    
    section_content = content[section_name]
    if section_name == 'decoder':
        isValid, error = verify_structure(section_content, decoder_struct)
        if not isValid:
            raise ValueError(f"Invalid decoder configuration in the config file, Error: {error}")
    elif section_name == 'trainer':
        isValid_nn, error_nn = verify_structure(section_content, trainer_struct_nn)
        isValid_linear, error_linear = verify_structure(section_content, trainer_struct_linear)
        if not (isValid_nn or isValid_linear):
            raise ValueError(f"Invalid trainer configuration in the config file, Error: {error_nn, error_linear}")
    elif section_name == 'preprocessing':
        for key in section_content:
            content = section_content[key] 
            isValid, error = verify_structure(content, preprocessing_struct)
        if not isValid:
                raise ValueError(f"Invalid preprocessor configuration in the config file, Error: {error}")
    else:
        raise ValueError(f"Unsupported section '{section_name}' in the config file.")
    
    return section_content

def compare_configs(current, original):
    differences = {}
    
    def recursive_compare(curr, orig, path=""):
        # Handle case where one is None and other isn't
        if curr is None and orig is not None:
            differences[path] = (orig, curr)
            return
        elif curr is not None and orig is None:
            differences[path] = (orig, curr)
            return
        elif curr is None and orig is None:
            return
            
        # Handle different types
        if type(curr) != type(orig):
            differences[path] = (orig, curr)
            return
            
        # Handle DictConfig/dict
        if isinstance(curr, (DictConfig, dict)) and isinstance(orig, (DictConfig, dict)):
            all_keys = set(curr.keys()) | set(orig.keys())
            for key in all_keys:
                new_path = f"{path}.{key}" if path else key
                curr_val = curr.get(key) if key in curr else None
                orig_val = orig.get(key) if key in orig else None
                recursive_compare(curr_val, orig_val, new_path)
                
        # Handle ListConfig/list
        elif isinstance(curr, (ListConfig, list)) and isinstance(orig, (ListConfig, list)):
            max_len = max(len(curr), len(orig))
            for i in range(max_len):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                curr_val = curr[i] if i < len(curr) else None
                orig_val = orig[i] if i < len(orig) else None
                recursive_compare(curr_val, orig_val, new_path)
                
        # Handle primitive values
        else:
            if curr != orig:
                differences[path] = (orig, curr)
    
    recursive_compare(current, original)
    return differences

if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../configs/parsing_test_1"):
        cfg = compose("config")

    decoder_config = parse_verify_config(cfg, 'decoder')
    print(decoder_config)
    trainer_config = parse_verify_config(cfg, 'trainer')
    print(trainer_config)