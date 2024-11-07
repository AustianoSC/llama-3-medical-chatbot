import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel
from peft.mixed_model import PeftMixedModel
from trl import setup_chat_format

from models.AppConfig import AppConfig

def _move_model_to_device(model: PreTrainedModel) -> PreTrainedModel:
    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

def initialize_model_quantized(config: AppConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
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
    _move_model_to_device(model)
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer

def initialize_model_for_merge(config: AppConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_config = config.model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.path,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=getattr(torch, model_config.torch_dtype),
        device_map="auto",
        trust_remote_code=True
    )
    _move_model_to_device(model)
    tokenizer = AutoTokenizer.from_pretrained(model_config.path)
    model, tokenizer = setup_chat_format(model, tokenizer)
    return model, tokenizer

def apply_peft_to_model(model: PreTrainedModel, config: AppConfig) -> (PeftModel | PeftMixedModel):
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

def merge_base_model_with_adapter(base_model: PreTrainedModel, adapter_model_path: str) -> PeftModel:
    model = PeftModel.from_pretrained(base_model, adapter_model_path)
    return model.merge_and_unload()