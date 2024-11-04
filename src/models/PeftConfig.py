from pydantic import BaseModel, Field
from typing import Optional

class PeftConfig(BaseModel):
    r: Optional[int] = Field(16, title="R", description="The number of random features to use for PEFT.")
    lora_alpha: Optional[float] = Field(32, title="LoRA Alpha", description="The alpha value to use for LoRA.")
    lora_dropout: Optional[float] = Field(0.05, title="LoRA Dropout", description="The dropout value to use for LoRA.")
    bias: Optional[str] = Field("none", title="Bias", description="Whether to use bias for PEFT.")
    task_type: Optional[str] = Field("lm", title="Task Type", description="The task type to use for PEFT.")
    target_modules: Optional[list] = Field(['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'], title="Target Modules", description="The target modules to apply PEFT to.")