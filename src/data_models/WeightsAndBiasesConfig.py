from pydantic import BaseModel, Field
from typing import Optional

class WeightsAndBiasesConfig(BaseModel):
    project_name: str = Field(..., title="Project Name", description="The name of the project to log training metrics to on Weights and Biases.")
    job_type: Optional[str] = Field("training", title="Job Type", description="The type of job to run on Weights and Biases.")
    anonymous: Optional[str] = Field("allow", title="Anonymous", description="Whether to log the training metrics anonymously on Weights and Biases.")