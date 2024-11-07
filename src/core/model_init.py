import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from trl import setup_chat_format

from models.AppConfig import AppConfig

def initialize_model(config: AppConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_config = config.model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_compute_dtype=getattr(torch, model_config.torch_dtype),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=model_config.attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer

def apply_peft(model: PreTrainedModel, config: AppConfig) -> (PeftModel | PeftMixedModel):
    peft_config = LoraConfig(
        r=config.peft.r,
        lora_alpha=config.peft.lora_alpha,
        lora_dropout=config.peft.lora_dropout,
        bias=config.peft.bias,
        task_type=config.peft.task_type,
        target_modules=config.peft.target_modules
    )
    model = get_peft_model(model, peft_config)
    return model