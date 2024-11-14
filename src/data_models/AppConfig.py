from pydantic import BaseModel, Field
from .ModelConfig import ModelConfig
from .DatasetConfig import DatasetConfig
from .TrainingConfig import TrainingConfig
from .PeftConfig import PeftConfig
from .EnvironmentVariables import EnvironmentVariables
from .WeightsAndBiasesConfig import WeightsAndBiasesConfig

class AppConfig(BaseModel):
    model: ModelConfig = Field(..., title="Model Configuration", description="The name of the datasets repo on HuggingFace or path to the datasets local directory.")
    dataset: DatasetConfig = Field(..., title="Dataset Configuration", description="The configuration for the dataset.")
    training: TrainingConfig = Field(..., title="Training Configuration", description="The configuration for the training process.")
    peft: PeftConfig = Field(..., title="PEFT Configuration", description="The configuration for the PEFT process.")
    env_vars: EnvironmentVariables = Field(..., title="Environment Variables", description="The environment variables to be set before running the training process.")
    weights_and_biases: WeightsAndBiasesConfig = Field(..., title="Weights and Biases Project Configuration", description="The configuration for the Weights and Biases project and logging.")