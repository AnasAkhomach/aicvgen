"""Pytest configuration and fixtures for CV Generator tests.

Provides shared fixtures, markers, and configuration for both
unit and integration tests.
"""

import pytest
import tempfile
import shutil
import os
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, Generator
from datetime import datetime

# Add project root to path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.state_manager import JobDescriptionData, WorkflowStage
from src.services.rate_limiter import RateLimitConfig
from src.services.session_manager import SessionManager
from src.core.state_manager import StateManager

# Try to import from models if available, otherwise use core
try:
    from src.models.data_models import (
        CVGenerationState, ContentItem, ContentType, 
        ProcessingMetadata, ProcessingStatus
    )
except ImportError:
    # Define minimal versions for testing if models don't exist
    from enum import Enum
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any, Optional
    
    class ContentType(Enum):
        EXPERIENCE = "experience"
        EDUCATION = "education"
        SKILLS = "skills"
        PROJECTS = "projects"
    
    class ProcessingStatus(Enum):
        PENDING = "pending"
        PROCESSING = "processing"
        COMPLETED = "completed"
        FAILED = "failed"
    
    @dataclass
    class ProcessingMetadata:
        item_id: str
        status: ProcessingStatus
        created_at: Optional[datetime] = None
        updated_at: Optional[datetime] = None
    
    @dataclass
    class ContentItem:
        content_type: ContentType
        original_content: str
        generated_content: Optional[str] = None
        metadata: Optional[ProcessingMetadata] = None
    
    @dataclass
    class CVGenerationState:
        session_id: str
        job_description: JobDescriptionData
        cv_data: Dict[str, Any]
        current_stage: WorkflowStage
        created_at: datetime
        updated_at: datetime


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring LLM service"
    )
    config.addinivalue_line(
        "markers", "requires_persistence: mark test as requiring persistence services"
    )


# Async test support
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Temporary directory fixtures
@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def session_storage_dir(temp_dir: str) -> str:
    """Provide a temporary directory for session storage."""
    session_dir = os.path.join(temp_dir, "sessions")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


@pytest.fixture
def state_storage_dir(temp_dir: str) -> str:
    """Provide a temporary directory for state storage."""
    state_dir = os.path.join(temp_dir, "state")
    os.makedirs(state_dir, exist_ok=True)
    return state_dir


# Mock fixtures
@pytest.fixture
def mock_llm_client():
    """Provide a mock LLM client for testing."""
    client = Mock()
    client.generate_content = AsyncMock()
    
    # Default successful response
    client.generate_content.return_value = {
        "content": "Generated test content",
        "tokens_used": 25,
        "model": "gpt-4"
    }
    
    return client


@pytest.fixture
def mock_progress_tracker():
    """Provide a mock progress tracker for testing."""
    tracker = Mock()
    tracker.update_progress = Mock()
    tracker.get_progress = Mock(return_value=0.5)
    tracker.reset_progress = Mock()
    return tracker


@pytest.fixture
def mock_error_recovery():
    """Provide a mock error recovery service for testing."""
    recovery = Mock()
    recovery.should_retry = Mock(return_value=True)
    recovery.get_retry_delay = Mock(return_value=0.1)
    recovery.get_backoff_delay = Mock(return_value=0.1)
    recovery.max_retries = 3
    recovery.base_delay = 0.1
    return recovery


# Configuration fixtures
@pytest.fixture
def rate_limit_config() -> RateLimitConfig:
    """Provide a test rate limit configuration."""
    return RateLimitConfig(
        requests_per_minute=10,
        requests_per_hour=100,
        tokens_per_minute=1000,
        tokens_per_hour=10000,
        base_backoff_seconds=1.0,
        max_backoff_seconds=60.0,
        jitter_enabled=True
    )


# Data model fixtures
@pytest.fixture
def sample_job_description() -> JobDescriptionData:
    """Provide a sample job description for testing."""
    return JobDescriptionData(
        title="Senior Software Engineer",
        company="TechCorp Inc",
        description="We are seeking a senior software engineer with expertise in Python and web development...",
        requirements=[
            "5+ years of Python development experience",
            "Experience with Django or Flask",
            "Knowledge of RESTful APIs",
            "Familiarity with cloud platforms (AWS, GCP, Azure)"
        ],
        responsibilities=[
            "Design and develop scalable web applications",
            "Collaborate with cross-functional teams",
            "Mentor junior developers",
            "Participate in code reviews"
        ],
        skills=[
            "Python", "Django", "Flask", "PostgreSQL", "Redis",
            "AWS", "Docker", "Git", "REST APIs", "JavaScript"
        ]
    )


@pytest.fixture
def sample_cv_data() -> Dict[str, Any]:
    """Provide sample CV data for testing."""
    return {
        "personal_info": {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+1-555-0123",
            "location": "San Francisco, CA",
            "linkedin": "https://linkedin.com/in/johndoe",
            "github": "https://github.com/johndoe"
        },
        "experience": [
            {
                "title": "Software Engineer",
                "company": "Previous Corp",
                "duration": "2020-2023",
                "location": "San Francisco, CA",
                "description": "Developed and maintained web applications using Python and Django. Collaborated with product teams to deliver features for 100k+ users."
            },
            {
                "title": "Junior Developer",
                "company": "StartupXYZ",
                "duration": "2018-2020",
                "location": "Palo Alto, CA",
                "description": "Built RESTful APIs and frontend components. Participated in agile development processes."
            }
        ],
        "projects": [
            {
                "name": "E-commerce Platform",
                "description": "Built a scalable e-commerce platform serving 10k+ daily users",
                "technologies": ["Python", "Django", "PostgreSQL", "Redis", "AWS"],
                "url": "https://github.com/johndoe/ecommerce"
            },
            {
                "name": "Analytics Dashboard",
                "description": "Real-time analytics dashboard with interactive visualizations",
                "technologies": ["React", "Node.js", "MongoDB", "D3.js"],
                "url": "https://github.com/johndoe/analytics"
            }
        ],
        "education": [
            {
                "degree": "Bachelor of Science in Computer Science",
                "institution": "University of California, Berkeley",
                "year": "2018",
                "gpa": "3.8"
            }
        ],
        "skills": {
            "programming_languages": ["Python", "JavaScript", "Java", "SQL"],
            "frameworks": ["Django", "Flask", "React", "Node.js"],
            "databases": ["PostgreSQL", "MongoDB", "Redis"],
            "tools": ["Git", "Docker", "AWS", "Jenkins"]
        },
        "certifications": [
            {
                "name": "AWS Certified Developer",
                "issuer": "Amazon Web Services",
                "year": "2022"
            }
        ]
    }


@pytest.fixture
def sample_content_item() -> ContentItem:
    """Provide a sample content item for testing."""
    return ContentItem(
        content_type=ContentType.EXPERIENCE,
        original_content="Software Engineer at Previous Corp (2020-2023)",
        metadata=ProcessingMetadata(
            item_id="exp-001",
            status=ProcessingStatus.PENDING
        )
    )


@pytest.fixture
def sample_cv_generation_state(sample_job_description: JobDescriptionData, sample_cv_data: Dict[str, Any]) -> CVGenerationState:
    """Provide a sample CV generation state for testing."""
    return CVGenerationState(
        session_id="test-session-001",
        job_description=sample_job_description,
        cv_data=sample_cv_data,
        current_stage=WorkflowStage.INITIALIZATION,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )


# Service fixtures
@pytest.fixture
def session_manager(session_storage_dir: str) -> SessionManager:
    """Provide a session manager with temporary storage."""
    return SessionManager(storage_path=session_storage_dir)


@pytest.fixture
def state_manager(state_storage_dir: str) -> StateManager:
    """Provide a state manager with temporary storage."""
    return StateManager(storage_path=state_storage_dir)


# Progress tracking fixture
@pytest.fixture
def progress_updates() -> list:
    """Provide a list to capture progress updates during testing."""
    return []


@pytest.fixture
def progress_callback(progress_updates: list):
    """Provide a progress callback that captures updates."""
    def callback(update: Dict[str, Any]):
        progress_updates.append(update)
    return callback


# Test data generators
@pytest.fixture
def content_item_factory():
    """Provide a factory for creating content items."""
    def create_content_item(
        content_type: ContentType = ContentType.EXPERIENCE,
        original_content: str = "Test content",
        generated_content: str = None,
        status: ProcessingStatus = ProcessingStatus.PENDING,
        item_id: str = None
    ) -> ContentItem:
        return ContentItem(
            content_type=content_type,
            original_content=original_content,
            generated_content=generated_content,
            metadata=ProcessingMetadata(
                item_id=item_id or f"{content_type.value}-test",
                status=status
            )
        )
    return create_content_item


@pytest.fixture
def job_description_factory():
    """Provide a factory for creating job descriptions."""
    def create_job_description(
        title: str = "Software Engineer",
        company: str = "Test Company",
        description: str = "Test job description",
        requirements: list = None,
        responsibilities: list = None,
        skills: list = None
    ) -> JobDescriptionData:
        return JobDescriptionData(
            title=title,
            company=company,
            description=description,
            requirements=requirements or ["Test requirement"],
            responsibilities=responsibilities or ["Test responsibility"],
            skills=skills or ["Test skill"]
        )
    return create_job_description


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_async_tasks():
    """Automatically cleanup async tasks after each test."""
    yield
    # Cancel any remaining tasks
    try:
        loop = asyncio.get_event_loop()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            if not task.done():
                task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception:
        pass  # Ignore cleanup errors


# Performance testing fixtures
@pytest.fixture
def performance_timer():
    """Provide a timer for performance testing."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self) -> float:
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return 0.0
    
    return Timer()


# Integration test specific fixtures
@pytest.fixture
def integration_test_config():
    """Provide configuration for integration tests."""
    return {
        "timeout": 30.0,  # Default timeout for async operations
        "max_retries": 3,
        "retry_delay": 0.1,
        "rate_limit_requests_per_minute": 10,
        "rate_limit_tokens_per_minute": 1000
    }


# Mock response generators
@pytest.fixture
def llm_response_generator():
    """Provide a generator for LLM responses."""
    def generate_response(
        content: str = "Generated content",
        tokens_used: int = 25,
        model: str = "gpt-4",
        delay: float = 0.0
    ):
        async def response(*args, **kwargs):
            if delay > 0:
                await asyncio.sleep(delay)
            return {
                "content": content,
                "tokens_used": tokens_used,
                "model": model
            }
        return response
    
    return generate_response