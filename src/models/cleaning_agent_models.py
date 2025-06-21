from typing import Optional
from pydantic import BaseModel, Field
from ..models.data_models import ProcessingStatus


class CleaningAgentNodeResult(BaseModel):
    cleaned_data: dict = Field(default_factory=dict)
    processing_status: ProcessingStatus = ProcessingStatus.FAILED
