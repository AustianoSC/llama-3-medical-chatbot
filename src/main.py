import wandb
import huggingface_hub as hf_hub

from utils.config import load_config
from core.model_init import initialize_model, apply_peft
from core.datasets import prepare_dataset
from core.train import train_model

def main(config_path):
    # Load environment variables and configs
    config = load_config(config_path)

    # HuggingFace and W&B login
    hf_hub.login(config.env_vars.huggingface_api_token)
    wandb.login(key=config.env_vars.wandb_api_key)

    # Initialize model and tokenizer
    model, tokenizer = initialize_model(config)
    
    # Apply PEFT
    model = apply_peft(model, config)
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    
    # Train model
    trainer = train_model(model, tokenizer, dataset, config)
    
    # Save and push the adapter model
    trainer.model.save_pretrained(config.training.output_dir)
    trainer.model.push_to_hub(config.training.output_dir, use_temp_dir=False)
    wandb.finish()

if __name__ == "__main__":
    config_path = "configs/example.yaml"
    main(config_path)