"""Configuration and fixtures for End-to-End tests.

Provides shared fixtures, test data, and configuration for E2E testing
of the CV tailoring application workflows.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import test data
from .test_data.sample_job_descriptions import SAMPLE_JOB_DESCRIPTIONS
from .test_data.sample_base_cvs import SAMPLE_BASE_CVS
from .test_data.mock_responses import MOCK_LLM_RESPONSES, MOCK_API_ERRORS
from .test_data.expected_outputs import (
    ExpectedCVOutputs, 
    CVQualityMetrics,
    get_expected_output_by_section,
    validate_cv_section_quality
)

# Import application components
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.orchestration.enhanced_orchestrator import EnhancedOrchestrator
from src.data_models.cv_models import CVGenerationRequest, CVGenerationState
from src.models.data_models import Item, ProcessingMetadata, ContentType, ItemType, ItemStatus
from src.services.session_manager import SessionManager
from src.services.state_manager import StateManager
from src.services.progress_tracker import ProgressTracker
from src.services.error_recovery_service import ErrorRecoveryService
from src.agents.item_processor import ItemProcessor
from src.services.rate_limiter import RateLimiter
from src.services.llm_client import LLMClient


# E2E Test Markers
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.requires_llm
]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def e2e_temp_dir():
    """Create a temporary directory for E2E test files."""
    temp_dir = tempfile.mkdtemp(prefix="aicvgen_e2e_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def e2e_session_storage_dir(e2e_temp_dir):
    """Create session storage directory for E2E tests."""
    session_dir = e2e_temp_dir / "sessions"
    session_dir.mkdir(exist_ok=True)
    return session_dir


@pytest.fixture
def e2e_state_storage_dir(e2e_temp_dir):
    """Create state storage directory for E2E tests."""
    state_dir = e2e_temp_dir / "states"
    state_dir.mkdir(exist_ok=True)
    return state_dir


@pytest.fixture
def e2e_output_dir(e2e_temp_dir):
    """Create output directory for E2E test results."""
    output_dir = e2e_temp_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for E2E tests."""
    mock_client = AsyncMock(spec=LLMClient)
    
    async def mock_generate_response(prompt: str, **kwargs) -> str:
        """Generate mock responses based on prompt content."""
        prompt_lower = prompt.lower()
        
        if "professional summary" in prompt_lower:
            return MOCK_LLM_RESPONSES["professional_summary"]
        elif "experience" in prompt_lower and "bullet" in prompt_lower:
            return MOCK_LLM_RESPONSES["experience_bullets"]
        elif "technical skills" in prompt_lower or "skills" in prompt_lower:
            return MOCK_LLM_RESPONSES["technical_skills"]
        elif "project" in prompt_lower:
            return MOCK_LLM_RESPONSES["projects"]
        else:
            return MOCK_LLM_RESPONSES["generic_response"]
    
    mock_client.generate_response = mock_generate_response
    mock_client.is_available = AsyncMock(return_value=True)
    mock_client.get_rate_limit_status = AsyncMock(return_value={"remaining": 100, "reset_time": 3600})
    
    return mock_client


@pytest.fixture
def mock_progress_tracker():
    """Mock progress tracker for E2E tests."""
    mock_tracker = MagicMock(spec=ProgressTracker)
    mock_tracker.start_session = MagicMock()
    mock_tracker.update_progress = MagicMock()
    mock_tracker.complete_session = MagicMock()
    mock_tracker.get_progress = MagicMock(return_value={"completed": 0, "total": 0, "percentage": 0.0})
    return mock_tracker


@pytest.fixture
def mock_error_recovery_service():
    """Mock error recovery service for E2E tests."""
    mock_service = AsyncMock(spec=ErrorRecoveryService)
    mock_service.handle_error = AsyncMock(return_value=True)
    mock_service.should_retry = AsyncMock(return_value=True)
    mock_service.get_retry_delay = AsyncMock(return_value=1.0)
    return mock_service


@pytest.fixture
def rate_limiter():
    """Rate limiter for E2E tests with relaxed limits."""
    return RateLimiter(
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_limit=10
    )


@pytest.fixture
def session_manager(e2e_session_storage_dir):
    """Session manager for E2E tests."""
    return SessionManager(storage_path=e2e_session_storage_dir)


@pytest.fixture
def state_manager(e2e_state_storage_dir):
    """State manager for E2E tests."""
    return StateManager(storage_path=e2e_state_storage_dir)


@pytest.fixture
def item_processor(mock_llm_client, rate_limiter):
    """Item processor for E2E tests."""
    return ItemProcessor(
        llm_client=mock_llm_client,
        rate_limiter=rate_limiter
    )


@pytest.fixture
def enhanced_orchestrator(
    mock_llm_client,
    mock_progress_tracker,
    mock_error_recovery_service,
    rate_limiter,
    session_manager,
    state_manager,
    item_processor
):
    """Enhanced orchestrator for E2E tests."""
    return EnhancedOrchestrator(
        llm_client=mock_llm_client,
        progress_tracker=mock_progress_tracker,
        error_recovery_service=mock_error_recovery_service,
        rate_limiter=rate_limiter,
        session_manager=session_manager,
        state_manager=state_manager,
        item_processor=item_processor
    )


@pytest.fixture(params=["software_engineer", "ai_engineer", "data_scientist"])
def job_role(request):
    """Parametrized job role for testing different scenarios."""
    return request.param


@pytest.fixture
def sample_job_description(job_role):
    """Get sample job description for the specified role."""
    return SAMPLE_JOB_DESCRIPTIONS[job_role]


@pytest.fixture
def sample_base_cv(job_role):
    """Get sample base CV for the specified role."""
    # Map job roles to experience levels
    role_to_level = {
        "software_engineer": "mid_level",
        "ai_engineer": "senior",
        "data_scientist": "junior"
    }
    level = role_to_level.get(job_role, "mid_level")
    return SAMPLE_BASE_CVS[level]


@pytest.fixture
def cv_generation_request(sample_job_description, sample_base_cv):
    """Create CV generation request for E2E tests."""
    return CVGenerationRequest(
        job_description=sample_job_description,
        base_cv_content=sample_base_cv,
        user_preferences={
            "max_pages": 2,
            "include_projects": True,
            "emphasize_technical_skills": True,
            "target_ats_optimization": True
        },
        processing_options={
            "enable_rate_limiting": True,
            "max_retries": 3,
            "timeout_seconds": 300,
            "quality_threshold": 0.8
        }
    )


@pytest.fixture
def expected_cv_outputs(job_role):
    """Get expected CV outputs for validation."""
    return {
        "professional_summary": get_expected_output_by_section("professional_summary", job_role),
        "professional_experience": get_expected_output_by_section("professional_experience", job_role),
        "technical_skills": get_expected_output_by_section("technical_skills", job_role),
        "projects": get_expected_output_by_section("projects", job_role)
    }


@pytest.fixture
def quality_metrics():
    """Get quality metrics for CV validation."""
    return {
        "content_quality": CVQualityMetrics.get_content_quality_criteria(),
        "formatting_standards": CVQualityMetrics.get_formatting_standards(),
        "ats_compatibility": CVQualityMetrics.get_ats_compatibility_requirements()
    }


@pytest.fixture
def performance_timer():
    """Timer for measuring E2E test performance."""
    import time
    
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.checkpoints = {}
        
        def start(self):
            self.start_time = time.time()
            return self
        
        def checkpoint(self, name: str):
            if self.start_time is None:
                raise ValueError("Timer not started")
            self.checkpoints[name] = time.time() - self.start_time
        
        def stop(self):
            if self.start_time is None:
                raise ValueError("Timer not started")
            self.end_time = time.time()
            return self.end_time - self.start_time
        
        def get_duration(self) -> float:
            if self.start_time is None or self.end_time is None:
                raise ValueError("Timer not properly started/stopped")
            return self.end_time - self.start_time
        
        def get_checkpoint_duration(self, name: str) -> float:
            if name not in self.checkpoints:
                raise ValueError(f"Checkpoint '{name}' not found")
            return self.checkpoints[name]
    
    return PerformanceTimer()


@pytest.fixture
def api_error_simulator():
    """Simulator for API errors during E2E tests."""
    
    class APIErrorSimulator:
        def __init__(self):
            self.error_scenarios = MOCK_API_ERRORS
            self.active_scenario = None
        
        def activate_scenario(self, scenario_name: str):
            if scenario_name not in self.error_scenarios:
                raise ValueError(f"Unknown error scenario: {scenario_name}")
            self.active_scenario = scenario_name
        
        def deactivate(self):
            self.active_scenario = None
        
        def should_raise_error(self) -> bool:
            return self.active_scenario is not None
        
        def get_error(self) -> Exception:
            if self.active_scenario is None:
                return None
            
            error_config = self.error_scenarios[self.active_scenario]
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
    
    return APIErrorSimulator()


@pytest.fixture
def cv_validator():
    """CV content validator for E2E tests."""
    
    class CVValidator:
        def __init__(self):
            self.validation_results = []
        
        def validate_section(self, section_content: str, expected_section) -> Dict[str, Any]:
            """Validate a CV section against expected criteria."""
            result = validate_cv_section_quality(section_content, expected_section)
            self.validation_results.append(result)
            return result
        
        def validate_full_cv(self, cv_content: str, expected_outputs: Dict[str, Any]) -> Dict[str, Any]:
            """Validate the complete CV against all expected outputs."""
            overall_results = {
                "overall_passed": True,
                "overall_score": 0.0,
                "section_results": {},
                "summary": {
                    "total_sections": len(expected_outputs),
                    "passed_sections": 0,
                    "failed_sections": 0,
                    "average_score": 0.0
                }
            }
            
            total_score = 0.0
            
            for section_name, expected_section in expected_outputs.items():
                # Extract section content (simplified - in real implementation, use proper parsing)
                section_content = self._extract_section_content(cv_content, section_name)
                
                if section_content:
                    result = self.validate_section(section_content, expected_section)
                    overall_results["section_results"][section_name] = result
                    
                    if result["passed"]:
                        overall_results["summary"]["passed_sections"] += 1
                    else:
                        overall_results["summary"]["failed_sections"] += 1
                        overall_results["overall_passed"] = False
                    
                    total_score += result["score"]
                else:
                    overall_results["section_results"][section_name] = {
                        "passed": False,
                        "score": 0.0,
                        "issues": [f"Section '{section_name}' not found in CV"]
                    }
                    overall_results["overall_passed"] = False
                    overall_results["summary"]["failed_sections"] += 1
            
            overall_results["overall_score"] = total_score / len(expected_outputs) if expected_outputs else 0.0
            overall_results["summary"]["average_score"] = overall_results["overall_score"]
            
            return overall_results
        
        def _extract_section_content(self, cv_content: str, section_name: str) -> Optional[str]:
            """Extract content for a specific section from CV (simplified implementation)."""
            # This is a simplified implementation - in reality, you'd use proper CV parsing
            section_keywords = {
                "professional_summary": ["summary", "profile", "objective"],
                "professional_experience": ["experience", "employment", "work history"],
                "technical_skills": ["skills", "technical", "technologies"],
                "projects": ["projects", "portfolio", "key projects"]
            }
            
            keywords = section_keywords.get(section_name, [section_name])
            
            for keyword in keywords:
                if keyword.lower() in cv_content.lower():
                    # Return a portion of the CV content (simplified)
                    start_idx = cv_content.lower().find(keyword.lower())
                    # Get next 500 characters as section content
                    return cv_content[start_idx:start_idx + 500]
            
            return None
        
        def get_validation_summary(self) -> Dict[str, Any]:
            """Get summary of all validation results."""
            if not self.validation_results:
                return {"message": "No validations performed"}
            
            total_validations = len(self.validation_results)
            passed_validations = sum(1 for result in self.validation_results if result["passed"])
            average_score = sum(result["score"] for result in self.validation_results) / total_validations
            
            return {
                "total_validations": total_validations,
                "passed_validations": passed_validations,
                "failed_validations": total_validations - passed_validations,
                "success_rate": passed_validations / total_validations,
                "average_score": average_score,
                "all_results": self.validation_results
            }
    
    return CVValidator()


# E2E Test Configuration
E2E_TEST_CONFIG = {
    "timeouts": {
        "complete_cv_generation": 300,  # 5 minutes
        "individual_item_processing": 60,  # 1 minute
        "error_recovery": 120  # 2 minutes
    },
    "performance_thresholds": {
        "max_processing_time": 180,  # 3 minutes
        "max_memory_usage_mb": 512,
        "min_success_rate": 0.95
    },
    "quality_thresholds": {
        "min_content_score": 0.8,
        "min_formatting_score": 0.9,
        "min_ats_compatibility": 0.85
    },
    "retry_settings": {
        "max_retries": 3,
        "retry_delay": 2.0,
        "exponential_backoff": True
    }
}


@pytest.fixture
def e2e_config():
    """E2E test configuration."""
    return E2E_TEST_CONFIG