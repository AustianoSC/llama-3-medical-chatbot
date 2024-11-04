import os
import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from models.AppConfig import AppConfig

def get_required_env_var(env_var_name: str) -> str:
    env_var = os.getenv(env_var_name)
    if env_var is None:
        raise ValueError(f"Environment variable {env_var_name} is required.")
    return env_var

def load_config(config_path: str) -> AppConfig:
    load_dotenv()
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    try:
        config_data['env_vars'] = {k.lower(): get_required_env_var(k) for k in ["HUGGINGFACE_API_TOKEN", "WANDB_API_KEY"]}
        config = AppConfig(**config_data)
        return config
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
        raise e