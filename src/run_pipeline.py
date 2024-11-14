import wandb
import huggingface_hub as hf_hub

from .train import train_model
from .data import load_dataset
from .utils import clean_gpu_mem
from .models import initialize_model_quantized, initialize_model_for_merge, apply_peft_to_model, merge_base_model_with_adapter
from .data_models import AppConfig

def run_pipeline(config: AppConfig):
    # HuggingFace and W&B login
    hf_hub.login(config.env_vars.huggingface_api_token)
    wandb.login(key=config.env_vars.wandb_api_key)
    wandb.init(
        project=config.weights_and_biases.project_name, 
        job_type=config.weights_and_biases.job_type, 
        anonymous=config.weights_and_biases.anonymous
    )

    # Initialize model and tokenizer
    model, tokenizer = initialize_model_quantized(config.model)
    
    # Apply PEFT
    model = apply_peft_to_model(model, config.peft)
    
    # Prepare dataset
    dataset = load_dataset(config.dataset, tokenizer)
    
    # Train model
    trainer = train_model(model, tokenizer, dataset, config.training)
    
    # Save and push the adapter model
    trainer.model.save_pretrained(config.training.output_dir)
    trainer.model.push_to_hub(config.training.output_dir, use_temp_dir=False)
    wandb.finish()

    # Delete old model, tokenizer, dataset, and trainer so we can initialize base model/tokenizer again
    del model, tokenizer, dataset, trainer.model, trainer
    
    # Cleanup remaning tensors from PyTorch caching
    clean_gpu_mem()

    # Initialize model and tokenizer again so we can combine with adapter model
    model, tokenizer = initialize_model_for_merge(config)

    # Merge base model with adapter model
    model = merge_base_model_with_adapter(model, config.training.output_dir)

    # Save and push the merged model
    model.save_pretrained(config.training.output_dir)
    tokenizer.save_pretrained(config.training.output_dir)
    model.push_to_hub(config.training.output_dir, use_temp_dir=False)
    tokenizer.push_to_hub(config.training.output_dir, use_temp_dir=False)