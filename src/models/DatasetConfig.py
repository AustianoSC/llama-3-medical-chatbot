from pydantic import BaseModel, Field
from typing import Optional

class DatasetConfig(BaseModel):
    name: str = Field(..., title="Dataset Name", description="The name of the dataset.")
    split: Optional[float] = Field(0.2, title="Split", description="How much of the dataset to set as the test set.")
    shuffle_seed: Optional[int] = Field(65, title="Shuffle Seed", description="The seed to use for shuffling/sampling the dataset.")
    select_top_n: Optional[int] = Field(1000, title="Select Top N", description="The number of samples to select from the dataset.")