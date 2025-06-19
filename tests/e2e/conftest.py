"""E2E test configuration and fixtures.

Provides fixtures for end-to-end testing of the AI CV generation system
after retry consolidation and service simplification.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import os

from src.services.llm_service import EnhancedLLMService
from src.services.item_processor import ItemProcessor
from src.services.session_manager import SessionManager
from src.services.progress_tracker import ProgressTracker
from src.services.error_recovery import ErrorRecoveryService
from src.core.state_manager import StateManager
from src.models.data_models import Item, ProcessingStatus, ItemStatus
from src.services.llm_service import LLMResponse


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-gemini-api-key-12345"


@pytest.fixture
def mock_llm_client(mock_api_key):
    """Create a mock LLM client for testing."""
    with patch('src.services.llm_service.genai') as mock_genai:
        # Mock the genai module
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        
        # Create real service instance with mocked API
        service = EnhancedLLMService(api_key=mock_api_key)
        service.llm = mock_model
        
        # Default successful response
        mock_response = Mock()
        mock_response.text = "Generated content from LLM"
        service.llm.generate_content_async = AsyncMock(return_value=mock_response)
        
        yield service


@pytest.fixture
def mock_progress_tracker():
    """Create a mock progress tracker."""
    tracker = Mock(spec=ProgressTracker)
    tracker.start_job = AsyncMock()
    tracker.update_progress = AsyncMock()
    tracker.complete_job = AsyncMock()
    tracker.fail_job = AsyncMock()
    tracker.get_progress = Mock(return_value={"completed": 0, "total": 0, "percentage": 0})
    return tracker


@pytest.fixture
def mock_error_recovery_service():
    """Create a mock error recovery service."""
    service = Mock(spec=ErrorRecoveryService)
    service.handle_error = AsyncMock(return_value=True)
    service.should_retry = Mock(return_value=False)
    service.get_recovery_strategy = Mock(return_value="skip")
    return service


@pytest.fixture
def session_manager(temp_dir):
    """Create a session manager for testing."""
    return SessionManager(base_path=temp_dir)


@pytest.fixture
def state_manager(temp_dir):
    """Create a state manager for testing."""
    return StateManager(storage_path=temp_dir / "state")


@pytest.fixture
def item_processor(mock_llm_client):
    """Create an item processor for testing."""
    return ItemProcessor(llm_client=mock_llm_client)


@pytest.fixture(params=["software_engineer", "ai_engineer", "data_scientist"])
def job_role(request):
    """Parametrized job role for testing different scenarios."""
    return request.param


@pytest.fixture
def sample_job_description(job_role):
    """Create sample job descriptions for different roles."""
    job_descriptions = {
        "software_engineer": {
            "title": "Senior Software Engineer",
            "company": "TechCorp",
            "requirements": [
                "5+ years of software development experience",
                "Proficiency in Python, Java, or C++",
                "Experience with web frameworks (Django, Flask, Spring)",
                "Knowledge of databases (SQL, NoSQL)",
                "Familiarity with cloud platforms (AWS, GCP, Azure)"
            ],
            "preferred_skills": [
                "Microservices architecture",
                "CI/CD pipelines",
                "Docker and Kubernetes",
                "Agile development methodologies"
            ],
            "description": "We are looking for a senior software engineer to join our growing team..."
        },
        "ai_engineer": {
            "title": "AI/ML Engineer",
            "company": "AI Innovations",
            "requirements": [
                "3+ years of machine learning experience",
                "Proficiency in Python and ML libraries (TensorFlow, PyTorch, scikit-learn)",
                "Experience with data preprocessing and feature engineering",
                "Knowledge of deep learning architectures",
                "Familiarity with MLOps practices"
            ],
            "preferred_skills": [
                "Natural Language Processing",
                "Computer Vision",
                "Model deployment and monitoring",
                "Big data technologies (Spark, Hadoop)"
            ],
            "description": "Join our AI team to develop cutting-edge machine learning solutions..."
        },
        "data_scientist": {
            "title": "Senior Data Scientist",
            "company": "DataTech Solutions",
            "requirements": [
                "4+ years of data science experience",
                "Strong statistical analysis skills",
                "Proficiency in Python/R and SQL",
                "Experience with data visualization tools",
                "Knowledge of A/B testing and experimental design"
            ],
            "preferred_skills": [
                "Business intelligence tools",
                "Advanced analytics and modeling",
                "Communication and presentation skills",
                "Domain expertise in relevant industry"
            ],
            "description": "We're seeking a senior data scientist to drive data-driven insights..."
        }
    }
    return job_descriptions[job_role]


@pytest.fixture
def sample_cv_data():
    """Create sample CV data for testing."""
    return {
        "personal_info": {
            "name": "John Doe",
            "email": "john.doe@email.com",
            "phone": "+1-555-0123",
            "location": "San Francisco, CA",
            "linkedin": "linkedin.com/in/johndoe",
            "github": "github.com/johndoe"
        },
        "summary": "Experienced software engineer with 8+ years of full-stack development experience...",
        "qualifications": [
            {
                "skill": "Python",
                "level": "Expert",
                "years_experience": 6,
                "description": "Extensive experience in Python development including web frameworks, data analysis, and automation"
            },
            {
                "skill": "JavaScript",
                "level": "Advanced",
                "years_experience": 5,
                "description": "Proficient in modern JavaScript, React, Node.js, and TypeScript"
            },
            {
                "skill": "Cloud Platforms",
                "level": "Intermediate",
                "years_experience": 3,
                "description": "Experience with AWS services, Docker, and Kubernetes"
            }
        ],
        "experience": [
            {
                "company": "TechStartup Inc.",
                "position": "Senior Software Engineer",
                "duration": "2020-2024",
                "achievements": [
                    "Led development of microservices architecture serving 1M+ users",
                    "Improved system performance by 40% through optimization",
                    "Mentored 3 junior developers"
                ]
            },
            {
                "company": "WebDev Corp",
                "position": "Software Engineer",
                "duration": "2018-2020",
                "achievements": [
                    "Developed RESTful APIs using Django and PostgreSQL",
                    "Implemented CI/CD pipelines reducing deployment time by 60%",
                    "Collaborated with cross-functional teams on product features"
                ]
            }
        ],
        "projects": [
            {
                "name": "E-commerce Platform",
                "technologies": ["Python", "Django", "PostgreSQL", "Redis", "AWS"],
                "description": "Built a scalable e-commerce platform handling 10K+ daily transactions",
                "achievements": [
                    "Implemented real-time inventory management",
                    "Integrated multiple payment gateways",
                    "Achieved 99.9% uptime"
                ]
            },
            {
                "name": "Data Analytics Dashboard",
                "technologies": ["React", "D3.js", "Python", "Flask", "MongoDB"],
                "description": "Created interactive dashboard for business intelligence",
                "achievements": [
                    "Processed and visualized 1TB+ of data",
                    "Reduced report generation time by 80%",
                    "Enabled real-time decision making"
                ]
            }
        ],
        "education": [
            {
                "degree": "Bachelor of Science in Computer Science",
                "institution": "University of Technology",
                "year": "2018",
                "gpa": "3.8/4.0"
            }
        ]
    }


@pytest.fixture
def sample_items(sample_cv_data):
    """Create sample items for processing."""
    items = []
    
    # Create qualification items
    for i, qual in enumerate(sample_cv_data["qualifications"]):
        items.append(Item(
            content=qual["description"],
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": f"qual-{i+1}",
                "item_type": "qualification",
                **qual
            }
        ))
    
    # Create experience items
    for i, exp in enumerate(sample_cv_data["experience"]):
        items.append(Item(
            content=f"{exp['position']} at {exp['company']}",
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": f"exp-{i+1}",
                "item_type": "experience",
                **exp
            }
        ))
    
    # Create project items
    for i, proj in enumerate(sample_cv_data["projects"]):
        items.append(Item(
            content=proj["description"],
            status=ItemStatus.INITIAL,
            metadata={
                "item_id": f"proj-{i+1}",
                "item_type": "project",
                **proj
            }
        ))
    
    return items


@pytest.fixture
def mock_successful_llm_response():
    """Create a mock successful LLM response."""
    return LLMResponse(
        success=True,
        content="Enhanced and tailored content generated by LLM",
        metadata={
            "model": "gemini-pro",
            "processing_time": 1.5,
            "tokens_used": 150
        }
    )


@pytest.fixture
def mock_failed_llm_response():
    """Create a mock failed LLM response."""
    return LLMResponse(
        success=False,
        content=None,
        error_message="LLM service failed after retries",
        metadata={
            "retry_count": 3,
            "last_error": "Rate limit exceeded"
        }
    )


@pytest.fixture
def environment_setup(monkeypatch, mock_api_key, temp_dir):
    """Set up test environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", mock_api_key)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CACHE_DIR", str(temp_dir / "cache"))
    monkeypatch.setenv("OUTPUT_DIR", str(temp_dir / "output"))
    
    # Create necessary directories
    (temp_dir / "cache").mkdir(exist_ok=True)
    (temp_dir / "output").mkdir(exist_ok=True)
    (temp_dir / "logs").mkdir(exist_ok=True)
    
    yield


@pytest.fixture
def qa_callback():
    """Create a mock QA callback for testing."""
    callback = Mock()
    callback.on_item_processed = AsyncMock()
    callback.on_quality_check = AsyncMock(return_value=True)
    callback.on_validation_failed = AsyncMock()
    return callback


# Utility fixtures for common test scenarios

@pytest.fixture
def processing_context(sample_job_description, environment_setup):
    """Create a complete processing context for E2E tests."""
    return {
        "job_description": sample_job_description,
        "session_id": "test-session-123",
        "trace_id": "test-trace-456",
        "user_preferences": {
            "style": "professional",
            "length": "detailed",
            "focus_areas": ["technical_skills", "leadership"]
        }
    }


@pytest.fixture
def error_scenarios():
    """Define various error scenarios for testing."""
    return {
        "rate_limit": {
            "exception": "google.api_core.exceptions.ResourceExhausted",
            "message": "Rate limit exceeded",
            "should_retry": True
        },
        "network_timeout": {
            "exception": "TimeoutError",
            "message": "Request timeout",
            "should_retry": True
        },
        "invalid_api_key": {
            "exception": "ConfigurationError",
            "message": "Invalid API key",
            "should_retry": False
        },
        "service_unavailable": {
            "exception": "google.api_core.exceptions.ServiceUnavailable",
            "message": "Service temporarily unavailable",
            "should_retry": True
        }
    }