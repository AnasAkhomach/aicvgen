from typing import Optional
from pydantic import BaseModel, Field

# DEPRECATED: Use FormatterAgentOutput from agent_output_models.py instead.
# class FormatterAgentNodeResult(BaseModel):
#     final_output_path: Optional[str] = None
#     error_messages: Optional[list] = Field(default_factory=list)
