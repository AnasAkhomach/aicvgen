from pydantic import BaseModel


class FormatterAgentOutput(BaseModel):
    formatted_cv_text: str
