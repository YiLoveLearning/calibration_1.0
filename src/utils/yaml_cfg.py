"""
This module provides utilities for loading, merging, saving and 
creating YAML configuration files.
"""

import copy
from collections import ChainMap
from pathlib import Path
import yaml
from loguru import logger

def merge_yaml(config_path_list):
    """Merge YAML configuration files from multiple yaml file.
    Args:
        config_path_list (list): A list of yaml file paths.
    
    Returns:
        dict: The merged configuration dictionary.
    """
    config_dict = {}

    for filename in config_path_list:
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            with open(filename, "r", encoding='utf-8') as file:
                config = yaml.safe_load(file)
                config_dict = dict(ChainMap(config, config_dict))

    return config_dict

def load_yaml(config_path):
    """Load a YAML configuration file.
    Args:
        config_path (str): The path to the YAML configuration file.
        
    Returns:
        dict: The configuration dictionary.
    
    Examples:
    >>> config_dict = load_yaml('../config/sumo_template.yaml')
    >>> print(config_dict)
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def save_yaml(config_dict, output_path):  # 写的不够好
    """Save a dictionary to a YAML file.
    Args:
        config_dict (dict): The configuration dictionary.
        output_path (str): The path to the output YAML file.
    
    Examples:
    >>> config_dict = {'a': 1, 'b': 2}
    >>> save_yaml(config_dict, 'output.yaml')
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        yaml.dump(config_dict, file)

# !灵活度较高!确保变量名和yaml中要改或增加的相同
def gen_yaml(config_dict, **kwargs):
    """Generate a YAML configuration file from a dictionary and keyword arguments.
    Args:
        config_dict (dict): The configuration dictionary.
        **kwargs: The keyword arguments to be added to the dictionary.
    
    Returns:
        dict: The updated configuration dictionary.
    
    Examples:
    >>> config_dict = {'a': 1, 'b': 2}
    >>> config_dict = gen_yaml(config_dict, c=3, d=4)
    >>> print(config_dict)

    >>> config_dict = {'a': 1, 'b': 2}
    >>> d = {'c': 3, 'd': 4}
    >>> config_dict = gen_yaml(config_dict, **d)
    """
    args_dict = locals()['kwargs']
    config_dict = copy.deepcopy(config_dict)
    for k, v in args_dict.items():
        if k not in config_dict:
            logger.info(f"New key-value pair {k}:{v} added to the dictionary.")
        config_dict[k] = v
    return config_dict

if __name__ == "__main__":
    yaml_list = [
        '../config/feedback_res_model_template.yaml',
        '../config/nn_model_template.yaml',
        '../config/sumo_template.yaml',
    ]
    print(merge_yaml(yaml_list))
