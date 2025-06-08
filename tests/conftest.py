#!/usr/bin/env python3
"""
Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests in the project.
It includes setup for mocking, test data, and environment configuration.
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Generator

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from src.config.environment import AppConfig, Environment, LoggingConfig, DatabaseConfig
from src.config.logging_config import setup_logging
from src.core.state_manager import StateManager, CVData, JobDescriptionData
from src.services.llm import LLM


@pytest.fixture(scope="session")
def test_config() -> AppConfig:
    """Provide test configuration for all tests."""
    config = AppConfig(
        environment=Environment.TESTING,
        debug=True,
        testing=True
    )
    
    # Override for testing
    config.logging.log_to_console = False
    config.logging.log_to_file = False
    config.database.backup_enabled = False
    config.performance.enable_caching = False
    
    return config


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_llm():
    """Provide a mocked LLM service."""
    with patch('src.services.llm.LLM') as mock:
        llm_instance = MagicMock()
        llm_instance.generate_content.return_value = "Mocked LLM response"
        llm_instance.generate_structured_content.return_value = {"key": "value"}
        mock.return_value = llm_instance
        yield llm_instance


@pytest.fixture
def mock_vector_db():
    """Provide a mocked vector database."""
    with patch('src.services.vector_db.VectorDB') as mock:
        db_instance = MagicMock()
        db_instance.search.return_value = [{"content": "test", "score": 0.9}]
        db_instance.add_documents.return_value = True
        mock.return_value = db_instance
        yield db_instance


@pytest.fixture
def sample_job_description() -> JobDescriptionData:
    """Provide sample job description data for testing."""
    return JobDescriptionData(
        raw_text="""Software Engineer Position
        
        We are looking for a skilled software engineer with experience in Python, 
        machine learning, and web development. The ideal candidate will have:
        
        - 3+ years of Python development experience
        - Experience with machine learning frameworks
        - Knowledge of web frameworks like Django or Flask
        - Strong problem-solving skills
        - Bachelor's degree in Computer Science or related field
        
        Responsibilities:
        - Develop and maintain software applications
        - Collaborate with cross-functional teams
        - Write clean, maintainable code
        - Participate in code reviews
        """,
        skills=["Python", "Machine Learning", "Web Development", "Django", "Flask"],
        experience_level="Mid-level (3+ years)",
        responsibilities=[
            "Develop and maintain software applications",
            "Collaborate with cross-functional teams",
            "Write clean, maintainable code",
            "Participate in code reviews"
        ],
        industry_terms=["Software Engineering", "Technology", "Development"],
        company_values=["Innovation", "Collaboration", "Quality"]
    )


@pytest.fixture
def sample_cv_data() -> CVData:
    """Provide sample CV data for testing."""
    return CVData(
        name="John Doe",
        email="john.doe@example.com",
        phone="+1-555-0123",
        linkedin="https://linkedin.com/in/johndoe",
        github="https://github.com/johndoe",
        summary="Experienced software engineer with 5+ years in Python development.",
        skills=[
            "Python", "JavaScript", "React", "Django", "PostgreSQL",
            "Machine Learning", "Docker", "AWS", "Git", "Agile"
        ],
        experience=[
            {
                "title": "Senior Software Engineer",
                "company": "Tech Corp",
                "duration": "2021-Present",
                "description": "Lead development of web applications using Python and React."
            },
            {
                "title": "Software Engineer",
                "company": "StartupXYZ",
                "duration": "2019-2021",
                "description": "Developed machine learning models and web APIs."
            }
        ],
        education=[
            {
                "degree": "Bachelor of Science in Computer Science",
                "institution": "University of Technology",
                "year": "2019"
            }
        ],
        projects=[
            {
                "name": "AI CV Generator",
                "description": "Built an AI-powered CV generation tool using Python and Streamlit.",
                "technologies": ["Python", "Streamlit", "OpenAI", "Docker"]
            }
        ],
        certifications=[
            "AWS Certified Developer",
            "Python Institute PCAP"
        ],
        languages=["English (Native)", "Spanish (Conversational)"]
    )


@pytest.fixture
def mock_state_manager(sample_cv_data, sample_job_description):
    """Provide a mocked state manager with sample data."""
    with patch('src.core.state_manager.StateManager') as mock:
        manager_instance = MagicMock()
        manager_instance.get_cv_data.return_value = sample_cv_data
        manager_instance.get_job_description.return_value = sample_job_description
        manager_instance.save_session.return_value = True
        manager_instance.load_session.return_value = True
        mock.return_value = manager_instance
        yield manager_instance


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit components for testing."""
    with patch('streamlit.write') as mock_write, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.warning') as mock_warning, \
         patch('streamlit.info') as mock_info, \
         patch('streamlit.success') as mock_success:
        
        yield {
            'write': mock_write,
            'error': mock_error,
            'warning': mock_warning,
            'info': mock_info,
            'success': mock_success
        }


@pytest.fixture
def mock_file_operations(temp_dir):
    """Mock file operations with temporary directory."""
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    
    # Create test directory structure
    (temp_dir / "data").mkdir(exist_ok=True)
    (temp_dir / "logs").mkdir(exist_ok=True)
    (temp_dir / "src" / "templates").mkdir(parents=True, exist_ok=True)
    
    yield temp_dir
    
    os.chdir(original_cwd)


@pytest.fixture
def sample_test_files(temp_dir):
    """Create sample test files."""
    files = {
        "cv.txt": "John Doe\nSoftware Engineer\nPython, JavaScript, React",
        "job_description.txt": "Looking for a Python developer with web experience",
        "template.md": "# {{name}}\n## Skills\n{{skills}}"
    }
    
    created_files = {}
    for filename, content in files.items():
        file_path = temp_dir / filename
        file_path.write_text(content, encoding='utf-8')
        created_files[filename] = file_path
    
    return created_files


@pytest.fixture(autouse=True)
def setup_test_environment(test_config):
    """Automatically set up test environment for all tests."""
    # Set environment variables for testing
    os.environ['APP_ENV'] = 'testing'
    os.environ['TESTING'] = 'true'
    os.environ['DEBUG'] = 'true'
    
    # Setup minimal logging for tests
    setup_logging(
        log_level=test_config.logging.get_log_level(),
        log_to_file=False,
        log_to_console=False
    )
    
    yield
    
    # Cleanup
    for key in ['APP_ENV', 'TESTING', 'DEBUG']:
        os.environ.pop(key, None)


@pytest.fixture
def mock_api_responses():
    """Provide mock API responses for external services."""
    return {
        "groq_success": {
            "choices": [{
                "message": {
                    "content": "This is a successful API response"
                }
            }]
        },
        "groq_error": {
            "error": {
                "message": "API rate limit exceeded",
                "type": "rate_limit_error"
            }
        }
    }


class TestHelpers:
    """Helper methods for testing."""
    
    @staticmethod
    def create_mock_response(content: str, status_code: int = 200):
        """Create a mock HTTP response."""
        mock_response = MagicMock()
        mock_response.text = content
        mock_response.status_code = status_code
        mock_response.json.return_value = {"content": content}
        return mock_response
    
    @staticmethod
    def assert_log_contains(caplog, level: str, message: str):
        """Assert that a log message was recorded."""
        for record in caplog.records:
            if record.levelname == level.upper() and message in record.message:
                return True
        pytest.fail(f"Log message '{message}' with level '{level}' not found")
    
    @staticmethod
    def create_test_session_data() -> Dict[str, Any]:
        """Create test session data."""
        return {
            "session_id": "test-session-123",
            "timestamp": "2024-01-01T00:00:00",
            "cv_data": {
                "name": "Test User",
                "email": "test@example.com"
            },
            "job_description": {
                "raw_text": "Test job description"
            }
        }


@pytest.fixture
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring API access"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add integration marker for tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add unit marker for tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)