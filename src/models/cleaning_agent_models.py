from typing import Optional, Any
from pydantic import BaseModel, Field
from ..models.data_models import ProcessingStatus


class CleanedDataModel(BaseModel):
    cleaned_data: Any
    confidence_score: float
    raw_output: Optional[str] = None
    output_type: Optional[str] = None


class CleaningAgentNodeResult(BaseModel):
    cleaned_data: dict = Field(default_factory=dict)
    processing_status: ProcessingStatus = ProcessingStatus.FAILED
