from pydantic import BaseModel, Field
from typing import Optional

class TrainingConfig(BaseModel):
    output_dir: str = Field(..., title="Output Directory", description="The local and HuggingFace directory to save the trained model.")
    batch_size: Optional[int] = Field(8, title="Batch Size", description="The batch size to use for training.")
    epochs: Optional[int] = Field(3, title="Epochs", description="The number of epochs to train the model.")
    learning_rate: Optional[float] = Field(1e-5, title="Learning Rate", description="The learning rate to use for training.")
    warmup_steps: Optional[int] = Field(1000, title="Warmup Steps", description="The number of warmup steps to use for training.")
    weight_decay: Optional[float] = Field(0.01, title="Weight Decay", description="The weight decay to use for training.")
    gradient_accumulation_steps: Optional[int] = Field(1, title="Gradient Accumulation Steps", description="The number of gradient accumulation steps to use for training.")
    optimizer: Optional[str] = Field("adamw_torch", title="Optimizer", description="The optimizer to use for training.")
    max_grad_norm: Optional[float] = Field(1.0, title="Max Grad Norm", description="The maximum gradient norm to use for training.")
    evaluation_strategy: Optional[str] = Field("steps", title="Evaluation Strategy", description="The evaluation strategy to use for training.")
    eval_steps: Optional[float] = Field(0.2, title="Eval Steps Interval", description="The interval to evaluate the model.")
    logging_steps: Optional[int] = Field(1, title="Log Steps Interval", description="The interval to log training metrics.")
    save_steps: Optional[int] = Field(1, title="Save Interval", description="The interval to save the model.")
    fp16: Optional[bool] = Field(False, title="FP16", description="Whether to use FP16 for training.")
    bf16: Optional[bool] = Field(False, title="BF16", description="Whether to use BF16 for training.")
    group_by_length: Optional[bool] = Field(False, title="Group by Length", description="Whether to group the dataset by length.")
    report_to: Optional[str] = Field("wandb", title="Report To", description="The service to report training metrics to.")