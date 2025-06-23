"""
llm_cv_parser_service.py
Service for LLM-based CV and job description parsing.
"""

from typing import Optional, Any
from src.models.data_models import JobDescriptionData, CVParsingResult


class LLMCVParserService:
    def __init__(self, llm_service, settings):
        self.llm_service = llm_service
        self.settings = settings

    async def parse_cv_with_llm(
        self,
        cv_text: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> CVParsingResult:
        prompt_path = self.settings.get_prompt_path_by_key("cv_parser")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = prompt_template.replace("{{raw_cv_text}}", cv_text)
        if self.llm_service is None:
            raise RuntimeError("LLM service is not available.")
        parsing_data = await self._generate_and_parse_json(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
        )
        return CVParsingResult(**parsing_data)

    async def parse_job_description_with_llm(
        self,
        raw_text: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> JobDescriptionData:
        prompt_path = self.settings.get_prompt_path_by_key("job_description_parser")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt = prompt_template.replace("{{raw_job_description}}", raw_text)
        if self.llm_service is None:
            raise RuntimeError("LLM service is not available.")
        parsing_data = await self._generate_and_parse_json(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
        )
        return JobDescriptionData(**parsing_data)

    async def _generate_and_parse_json(
        self, prompt: str, session_id: Optional[str], trace_id: Optional[str]
    ) -> Any:
        # Centralized LLM call and JSON parsing logic
        llm_response = await self.llm_service.generate(
            prompt=prompt,
            session_id=session_id,
            trace_id=trace_id,
        )
        # Assume llm_response is JSON or can be parsed as such
        import json

        return json.loads(llm_response)
