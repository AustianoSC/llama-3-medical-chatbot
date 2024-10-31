import os
import yaml
from dotenv import load_dotenv

def load_config(config_path):
    load_dotenv()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config