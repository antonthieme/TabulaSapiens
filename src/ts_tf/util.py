import argparse
import yaml
import json
import os

def load_config(config_path):
    """Load configuration from a YAML or JSON file."""
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            return yaml.safe_load(f)
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            raise ValueError("Unsupported config file format. Use YAML or JSON.")

def save_final_config(config, output_path):
    """Save the final configuration to a file."""
    with open(output_path, 'w') as f:
        yaml.dump(config, f)