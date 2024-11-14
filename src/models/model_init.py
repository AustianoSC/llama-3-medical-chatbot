import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from trl import setup_chat_format

from ..data_models import ModelConfig, PeftConfig

def initialize_model_quantized(model_config: ModelConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
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

def initialize_model_for_merge(model_config: ModelConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    model = AutoModelForCausalLM.from_pretrained(
            model_config.path,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=getattr(torch, model_config.torch_dtype),
            device_map="cpu",
            trust_remote_code=True
        )
    
    try:
        model.to("cuda")
    except RuntimeError as e:
        print(f'Running base model on CPU due to CUDA error: {e}')
    
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    return model, tokenizer

def apply_peft_to_model(model: PreTrainedModel, peft_config: PeftConfig) -> (PeftModel | PeftMixedModel):
    peft_model_config = LoraConfig(
        r=peft_config.r,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,
        bias=peft_config.bias,
        task_type=peft_config.task_type,
        target_modules=peft_config.target_modules
    )
    model = get_peft_model(model, peft_model_config)
    return model

def merge_base_model_with_adapter(base_model: PreTrainedModel, adapter_model_path: str) -> PeftModel:
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    return model.merge_and_unload()