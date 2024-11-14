import argparse
from src import run_pipeline, utils

def main():
    config_path = "configs/example.yaml"

    # Load environment variables and configs
    config = utils.load_config(config_path)

    run_pipeline(config)

if __name__ == "__main__":
    main()