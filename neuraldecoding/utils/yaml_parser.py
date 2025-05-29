import yaml
from typing import Any

def verify_structure(content: dict, struct: dict) -> tuple[bool, str]:
    """
    Recursively verify if the content dictionary matches the structure and types defined in struct.

    Parameters:
    content (dict): The content to verify.
    struct (dict): The structure definition where keys are expected keys and values are types or nested structures.

    Returns:
    (bool, str):bool: True if the content matches the structure, False otherwise.
                str: An error message if the content does not match the structure, None if it matches.
    """
    for key, value_type in struct.items():
        if key not in content:
            return False, f"Missing key: {key}"
        if isinstance(value_type, dict):
            if not isinstance(content[key], dict):
                return False, f"Key '{key}' is not a dictionary as expected."
            is_valid, error = verify_structure(content[key], value_type)
            if not is_valid:
                return False, error
        else:
            if not isinstance(content[key], value_type):
                return False, f"Key '{key}' has incorrect type. Expected {value_type}, got {type(content[key])}."
    return True, None

def verify_decoder(content:dict) -> tuple[bool,str]:
    """
    Verify if the content is a valid decoder configuration.

    Parameters:
    content (dict): The content to verify.

    Returns:
    (bool, str):bool: True if the content matches the structure, False otherwise.
                str: An error message if the content does not match the structure, None if it matches.
    """
    decoder_struct = {'model': {'name': str, 
                                 'parameters': dict, 
                                 'input_shape': list, 
                                 'output_shape': list}, 
                       'stabilization': {'name': str, 
                                         'parameters': dict, 
                                         'date_0': str, 
                                         'date_k': str},
                       'fpath': str
                    }
    return verify_structure(content, decoder_struct)

def load_yaml(file_path: str, section_name: str) -> dict:
    """
    Load a YAML file and return its content based on the type of class required.

    Parameters:
    file_path (str): The path to the YAML file.
    section_name (str): The section name to extract from the YAML file.

    Returns:
    dict: The content of the specified section in the YAML file.
    """
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
    
    if section_name not in content:
        raise ValueError(f"Section '{section_name}' not found in the YAML file.")
    
    section_content = content[section_name]

    if section_name == 'decoder':
        isValid, error = verify_decoder(section_content)
        if not isValid:
            raise ValueError(f"Invalid decoder configuration in the YAML file, Error: {error}")
    else:
        raise ValueError(f"Unsupported section '{section_name}' in the YAML file.")
    
    return section_content

if __name__ == "__main__":
    # Example usage
    try:
        decoder_config = load_yaml('configs/toplevelconfigtest.yaml', 'decoder')
        print("Decoder configuration loaded successfully:", decoder_config)
    except Exception as e:
        print("Error loading YAML file:", e)