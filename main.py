import argparse
from src import run_pipeline, utils

def main():
    parser = argparse.ArgumentParser(description="Run the medical chatbot pipeline.")
    parser.add_argument("--config", type=str, default="configs/example.yaml", help="Path to the config file.")
    args = parser.parse_args()

    config_path = args.config

    # Load environment variables and configs
    config = utils.load_config(config_path)

    run_pipeline(config)

if __name__ == "__main__":
    main()