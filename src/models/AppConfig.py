from pydantic import BaseModel, Field
from typing import Optional

class ModelConfig(BaseModel):
    path: str = Field(..., title="Model Path", description="The name of the model's repo on HuggingFace or the path to the model's local directory.")
    load_in_4bit: bool = Field(..., title="Load in 4-bit", description="Whether to load the model in 4-bit mode.")
    torch_dtype: str = Field(..., title="Torch Data Type", description="The torch data type to use for the model.")
    attn_implementation: str = Field(..., title="Attention Implementation", description="The attention implementation to use for the model.")

class DatasetConfig(BaseModel):
    name: str = Field(..., title="Dataset Name", description="The name of the dataset.")
    split: Optional[float] = Field(0.2, title="Split", description="How much of the dataset to set as the test set.")
    shuffle_seed: Optional[int] = Field(65, title="Shuffle Seed", description="The seed to use for shuffling/sampling the dataset.")
    select_top_n: Optional[int] = Field(1000, title="Select Top N", description="The number of samples to select from the dataset.")

class AppConfig(BaseModel):
    model: ModelConfig = Field(..., title="Model Configuration", description="The name of the datasets repo on HuggingFace or path to the datasets local directory.")
    dataset: DatasetConfig = Field(..., title="Dataset Configuration", description="The configuration for the dataset.")