from utils.config import load_config
from core.model_init import initialize_model, apply_peft
from core.datasets import prepare_dataset
from core.train import train_model

def main(config_path):
    # Load environment variables and configs
    config = load_config(config_path)
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model(config)
    
    # Apply PEFT
    model = apply_peft(model, config)
    
    # Prepare dataset
    dataset = prepare_dataset(config, tokenizer)
    
    # Train model
    trainer = train_model(model, tokenizer, dataset, config)
    
    # Save and push model
    trainer.model.save_pretrained(config['training']['output_dir'])
    trainer.model.push_to_hub(config['training']['output_dir'], use_temp_dir=False)

if __name__ == "__main__":
    config_path = "configs/example.yaml"
    main(config_path)