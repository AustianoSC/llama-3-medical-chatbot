from pydantic import BaseModel, Field

class EnvironmentVariables(BaseModel):
    huggingface_api_token: str = Field(..., title="HuggingFace API Token", description="The API Token for the HuggingFace account.")
    wandb_api_key: str = Field(..., title="WandB API Key", description="The API Key for the WandB account.")