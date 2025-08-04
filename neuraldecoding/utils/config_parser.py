from omegaconf import DictConfig, ListConfig
from neuraldecoding.utils.config_structs import decoder_struct, trainer_struct_nn, trainer_struct_linear, preprocessing_struct
from typing import Union, get_origin, get_args

def verify_structure(content: DictConfig, struct: dict) -> tuple[bool, str | None]:
    """
    Recursively verify if the content DictConfig matches the structure and types defined in struct.

    Parameters:
    content (DictConfig): The config content to validate.
    struct (dict): A nested structure definition with expected keys and types.

    Returns:
    (bool, str | None): (True, None) if valid; (False, error message) otherwise.
    """
    for key, expected_type in struct.items():
        if key not in content:
            return False, f"Missing key: '{key}'"

        val = content[key]

        # Nested dicts: recurse
        if isinstance(expected_type, dict):
            if not isinstance(val, (DictConfig, dict)):
                return False, f"Key '{key}' should be a dict or DictConfig."
            is_valid, error = verify_structure(val, expected_type)
            if not is_valid:
                return False, f"In key '{key}': {error}"

        # Handle Union[...] types
        else:
            origin = get_origin(expected_type)
            args = get_args(expected_type)

            if origin is Union:
                if not any(
                    (arg is type(None) and val is None) or isinstance(val, arg)
                    for arg in args
                ):
                    expected_names = ", ".join(str(a) for a in args)
                    return False, f"Key '{key}' has incorrect type. Expected one of [{expected_names}], got {type(val)}."
            else:
                if not isinstance(val, expected_type):
                    return False, f"Key '{key}' has incorrect type. Expected {expected_type}, got {type(val)}."

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

if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../configs/parsing_test_1"):
        cfg = compose("config")

    decoder_config = parse_verify_config(cfg, 'decoder')
    print(decoder_config)
    trainer_config = parse_verify_config(cfg, 'trainer')
    print(trainer_config)