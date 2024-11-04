from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    path: str = Field(..., title="Model Path", description="The name of the model's repo on HuggingFace or the path to the model's local directory.")
    load_in_4bit: bool = Field(..., title="Load in 4-bit", description="Whether to load the model in 4-bit mode.")
    torch_dtype: str = Field(..., title="Torch Data Type", description="The torch data type to use for the model.")
    attn_implementation: str = Field(..., title="Attention Implementation", description="The attention implementation to use for the model.")