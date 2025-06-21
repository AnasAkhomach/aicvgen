from typing import Optional
from pydantic import BaseModel, Field


class FormatterAgentNodeResult(BaseModel):
    final_output_path: Optional[str] = None
    error_messages: Optional[list] = Field(default_factory=list)
