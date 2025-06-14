"""Mock LLM service for deterministic E2E testing."""

import asyncio
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock
import json
import re

from .test_data.mock_responses import MOCK_LLM_RESPONSES, MOCK_API_ERRORS


class MockLLMService:
    """Mock LLM service that provides deterministic responses for testing."""
    
    def __init__(self, enable_errors: bool = False, error_rate: float = 0.0):
        self.enable_errors = enable_errors
        self.error_rate = error_rate
        self.call_count = 0
        self.call_history = []
        self.response_delay = 0.1  # Simulate network delay
        self.active_error_scenario = None
        
    async def generate_content_async(self, prompt: str, **kwargs) -> str:
        """Generate mock content based on prompt analysis."""
        await asyncio.sleep(self.response_delay)
        
        self.call_count += 1
        self.call_history.append({
            "prompt": prompt,
            "kwargs": kwargs,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Simulate errors if enabled
        if self.enable_errors and self._should_raise_error():
            raise self._get_mock_error()
        
        return self._generate_response_for_prompt(prompt)
    
    async def generate_structured_content_async(self, prompt: str, response_format: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Generate structured mock content."""
        await asyncio.sleep(self.response_delay)
        
        self.call_count += 1
        self.call_history.append({
            "prompt": prompt,
            "response_format": response_format,
            "kwargs": kwargs,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Simulate errors if enabled
        if self.enable_errors and self._should_raise_error():
            raise self._get_mock_error()
        
        return self._generate_structured_response_for_prompt(prompt, response_format)
    
    def _generate_response_for_prompt(self, prompt: str) -> str:
        """Generate appropriate mock response based on prompt content."""
        prompt_lower = prompt.lower()
        
        # Executive summary
        if any(keyword in prompt_lower for keyword in ["executive summary", "professional summary", "summary", "profile"]):
            return MOCK_LLM_RESPONSES["executive_summary"]
        
        # Experience bullets
        elif any(keyword in prompt_lower for keyword in ["experience", "bullet", "responsibilities", "achievements"]):
            return MOCK_LLM_RESPONSES["experience_bullets"]
        
        # Technical skills
        elif any(keyword in prompt_lower for keyword in ["technical skills", "skills", "technologies", "programming"]):
            return MOCK_LLM_RESPONSES["technical_skills"]
        
        # Projects
        elif any(keyword in prompt_lower for keyword in ["project", "portfolio", "key projects"]):
            return MOCK_LLM_RESPONSES["projects"]
        
        # Big 10 skills
        elif any(keyword in prompt_lower for keyword in ["big 10", "top 10", "most important skills"]):
            return MOCK_LLM_RESPONSES["big_10_skills"]["raw_output"]
        
        # Quality assurance
        elif any(keyword in prompt_lower for keyword in ["quality", "review", "feedback", "assess"]):
            return json.dumps(MOCK_LLM_RESPONSES["quality_assurance_feedback"])
        
        # Research
        elif any(keyword in prompt_lower for keyword in ["research", "company", "industry"]):
            return json.dumps(MOCK_LLM_RESPONSES["research_findings"])
        
        # Default response
        else:
            return MOCK_LLM_RESPONSES["generic_response"]
    
    def _generate_structured_response_for_prompt(self, prompt: str, response_format: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured mock response based on prompt and expected format."""
        prompt_lower = prompt.lower()
        
        # Big 10 skills structured response
        if any(keyword in prompt_lower for keyword in ["big 10", "top 10", "skills"]):
            return {
                "skills": MOCK_LLM_RESPONSES["big_10_skills"]["structured_skills"],
                "raw_output": MOCK_LLM_RESPONSES["big_10_skills"]["raw_output"]
            }
        
        # Quality assurance structured response
        elif any(keyword in prompt_lower for keyword in ["quality", "review", "feedback"]):
            return MOCK_LLM_RESPONSES["quality_assurance_feedback"]
        
        # Research structured response
        elif any(keyword in prompt_lower for keyword in ["research", "company", "industry"]):
            return MOCK_LLM_RESPONSES["research_findings"]
        
        # Default structured response
        else:
            return {
                "content": self._generate_response_for_prompt(prompt),
                "metadata": {
                    "generated_by": "mock_llm_service",
                    "prompt_type": "unknown",
                    "confidence": 0.85
                }
            }
    
    def _should_raise_error(self) -> bool:
        """Determine if an error should be raised based on error rate."""
        if self.active_error_scenario:
            return True
        
        import random
        return random.random() < self.error_rate
    
    def _get_mock_error(self) -> Exception:
        """Get appropriate mock error."""
        if self.active_error_scenario:
            error_config = MOCK_API_ERRORS[self.active_error_scenario]
        else:
            # Random error for general error simulation
            import random
            error_config = random.choice(list(MOCK_API_ERRORS.values()))
        
        error_type = error_config["type"]
        error_message = error_config["message"]
        
        if error_type == "rate_limit":
            from src.exceptions.rate_limit_exceptions import RateLimitExceededException
            return RateLimitExceededException(error_message)
        elif error_type == "timeout":
            return TimeoutError(error_message)
        elif error_type == "authentication":
            from src.exceptions.llm_exceptions import LLMAuthenticationError
            return LLMAuthenticationError(error_message)
        elif error_type == "api_error":
            from src.exceptions.llm_exceptions import LLMAPIError
            return LLMAPIError(error_message)
        else:
            return Exception(error_message)
    
    def activate_error_scenario(self, scenario_name: str):
        """Activate a specific error scenario."""
        if scenario_name not in MOCK_API_ERRORS:
            raise ValueError(f"Unknown error scenario: {scenario_name}")
        self.active_error_scenario = scenario_name
    
    def deactivate_error_scenario(self):
        """Deactivate error scenario."""
        self.active_error_scenario = None
    
    def reset_call_history(self):
        """Reset call history and counters."""
        self.call_count = 0
        self.call_history = []
    
    def get_call_count(self) -> int:
        """Get the total number of LLM calls made."""
        return self.call_count
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get the complete call history."""
        return self.call_history.copy()
    
    def get_call_statistics(self) -> Dict[str, Any]:
        """Get statistics about LLM calls."""
        if not self.call_history:
            return {
                "total_calls": 0,
                "average_response_time": 0.0,
                "prompt_types": {}
            }
        
        # Analyze prompt types
        prompt_types = {}
        for call in self.call_history:
            prompt = call["prompt"].lower()
            
            if "summary" in prompt:
                prompt_type = "executive_summary"
            elif "experience" in prompt:
                prompt_type = "experience"
            elif "skills" in prompt:
                prompt_type = "skills"
            elif "project" in prompt:
                prompt_type = "projects"
            elif "quality" in prompt:
                prompt_type = "quality_assurance"
            elif "research" in prompt:
                prompt_type = "research"
            else:
                prompt_type = "other"
            
            prompt_types[prompt_type] = prompt_types.get(prompt_type, 0) + 1
        
        return {
            "total_calls": self.call_count,
            "average_response_time": self.response_delay,
            "prompt_types": prompt_types,
            "error_rate": self.error_rate,
            "active_error_scenario": self.active_error_scenario
        }


class MockLLMServiceFactory:
    """Factory for creating configured mock LLM services."""
    
    @staticmethod
    def create_reliable_service() -> MockLLMService:
        """Create a reliable mock service with no errors."""
        return MockLLMService(enable_errors=False, error_rate=0.0)
    
    @staticmethod
    def create_unreliable_service(error_rate: float = 0.1) -> MockLLMService:
        """Create an unreliable mock service with random errors."""
        return MockLLMService(enable_errors=True, error_rate=error_rate)
    
    @staticmethod
    def create_slow_service(delay: float = 1.0) -> MockLLMService:
        """Create a slow mock service for performance testing."""
        service = MockLLMService(enable_errors=False, error_rate=0.0)
        service.response_delay = delay
        return service
    
    @staticmethod
    def create_error_scenario_service(scenario: str) -> MockLLMService:
        """Create a mock service with a specific error scenario."""
        service = MockLLMService(enable_errors=True, error_rate=1.0)
        service.activate_error_scenario(scenario)
        return service