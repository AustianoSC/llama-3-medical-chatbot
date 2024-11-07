import gc
import torch
import wandb
import huggingface_hub as hf_hub

from utils.config import load_config
from core.train import train_model
from core.datasets import prepare_dataset
from core.model_init import initialize_model_quantized, initialize_model_for_merge, apply_peft_to_model, merge_base_model_with_adapter

def main(config_path):
    # Load environment variables and configs
    config = load_config(config_path)

    # HuggingFace and W&B login
    hf_hub.login(config.env_vars.huggingface_api_token)
    wandb.login(key=config.env_vars.wandb_api_key)

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_quantized(config)
    
    # Apply PEFT
    model = apply_peft_to_model(model, config)
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    
    # Train model
    trainer = train_model(model, tokenizer, dataset, config)
    
    # Save and push the adapter model
    trainer.model.save_pretrained(config.training.output_dir)
    trainer.model.push_to_hub(config.training.output_dir, use_temp_dir=False)
    wandb.finish()

    # Delete old model, tokenizer, dataset, and trainer so we can initialize base model/tokenizer again
    del model, tokenizer, dataset, trainer.model, trainer
    
    # Cleanup remaning tensors from PyTorch caching
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                del obj
        except AttributeError:
            continue
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()

    # Initialize model and tokenizer again so we can combine with adapter model
    model, tokenizer = initialize_model_for_merge(config)

    # Merge base model with adapter model
    model = merge_base_model_with_adapter(model, config.training.output_dir)

    # Save and push the merged model
    model.save_pretrained(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    model.push_to_hub(config.training.output_dir, use_temp_dir=False)
    tokenizer.push_to_hub(config.training.output_dir, use_temp_dir=False)

if __name__ == "__main__":
    config_path = "configs/example.yaml"
    main(config_path)