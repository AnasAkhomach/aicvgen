# API Reference

## Overview

The AI CV Generator provides both a Streamlit web interface and programmatic access through its core modules. This document outlines the key APIs and interfaces available for developers.

## Core Classes and Interfaces

### EnhancedOrchestrator

The main orchestration class that manages the CV generation workflow.

```python
from src.core.orchestrator import EnhancedOrchestrator

orchestrator = EnhancedOrchestrator()
result = await orchestrator.process_cv_request(
    job_description="Software Engineer position...",
    base_cv_content="My CV content...",
    session_id="unique_session_id"
)
```

#### Methods

- `process_cv_request(job_description: str, base_cv_content: str, session_id: str) -> CVGenerationResult`
  - Main method for CV generation
  - Returns structured result with sections and items

- `regenerate_item(item_id: str, feedback: str = None) -> Item`
  - Regenerates a specific CV item with optional user feedback

- `get_session_state(session_id: str) -> SessionState`
  - Retrieves current session state

### Data Models

All data models are defined using Pydantic for type safety and validation.

#### Item

```python
from src.data.models import Item, ItemType, ItemStatus

item = Item(
    id="unique_id",
    content="Bullet point content",
    item_type=ItemType.BULLET_POINT,
    status=ItemStatus.PENDING,
    metadata={"confidence": 0.85}
)
```

#### Section

```python
from src.data.models import Section

section = Section(
    id="experience",
    title="Professional Experience",
    items=[item1, item2, item3],
    metadata={"priority": "high"}
)
```

#### CVGenerationResult

```python
from src.data.models import CVGenerationResult

result = CVGenerationResult(
    session_id="session_123",
    sections=[section1, section2],
    big_10_skills=["Python", "Machine Learning", ...],
    processing_time=25.3,
    metadata={"agent_responses": {...}}
)
```

### Agent Interfaces

All agents inherit from `BaseAgent` and implement specific functionality.

#### ContentWriterAgent

```python
from src.agents.content_writer import ContentWriterAgent

agent = ContentWriterAgent()
result = await agent.generate_content(
    job_description="Job requirements...",
    base_content="Original content...",
    context={"section": "experience"}
)
```

#### ResearchAgent

```python
from src.agents.research_agent import ResearchAgent

agent = ResearchAgent()
skills = await agent.extract_skills(job_description)
requirements = await agent.analyze_requirements(job_description)
```

#### QAAgent

```python
from src.agents.qa_agent import QAAgent

agent = QAAgent()
validation = await agent.validate_content(
    content="Generated content...",
    requirements=["requirement1", "requirement2"]
)
```

### Services

#### LLMService

```python
from src.services.llm_service import LLMService

llm = LLMService()
response = await llm.generate_content(
    prompt="Generate a professional summary...",
    context={"job_title": "Software Engineer"},
    max_tokens=500
)
```

#### SessionManager

```python
from src.core.session_manager import SessionManager

session_mgr = SessionManager()
session_mgr.create_session("user_123")
session_mgr.save_state(session_id, state_data)
state = session_mgr.load_state(session_id)
```

#### StateManager

```python
from src.core.state_manager import StateManager

state_mgr = StateManager()
state_mgr.update_item_status(item_id, ItemStatus.ACCEPTED)
state_mgr.add_user_feedback(item_id, "Please make this more specific")
```

## Configuration

### Environment Variables

```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional
LOG_LEVEL=INFO
SESSION_TIMEOUT=3600
MAX_CONCURRENT_REQUESTS=10
ENABLE_DEBUG_MODE=false
```

### Logging Configuration

Logging is configured via `config/logging.yaml`:

```yaml
version: 1
formatters:
  secure:
    format: '[{asctime}] {levelname} - {name} - {message}'
    style: '{'
handlers:
  file:
    class: logging.FileHandler
    filename: logs/app.log
    formatter: secure
loggers:
  aicvgen:
    level: INFO
    handlers: [file]
    propagate: false
```

## Error Handling

### Custom Exceptions

```python
from src.utils.exceptions import (
    CVGenerationError,
    LLMServiceError,
    SessionNotFoundError,
    ValidationError
)

try:
    result = await orchestrator.process_cv_request(...)
except LLMServiceError as e:
    # Handle LLM service failures
    logger.error(f"LLM service failed: {e}")
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Validation failed: {e}")
```

### Fallback Mechanisms

The system includes automatic fallbacks:

- **LLM Failures**: Falls back to template-based content
- **Network Issues**: Uses cached responses when available
- **Validation Errors**: Provides default content with warnings

## Performance Considerations

### Optimization Tips

1. **Session Reuse**: Reuse sessions for multiple operations
2. **Batch Processing**: Process multiple items together when possible
3. **Caching**: Enable response caching for repeated requests
4. **Async Operations**: Use async/await for all I/O operations

### Performance Metrics

```python
from src.utils.metrics import PerformanceTracker

tracker = PerformanceTracker()
with tracker.measure("cv_generation"):
    result = await orchestrator.process_cv_request(...)

metrics = tracker.get_metrics()
print(f"Processing time: {metrics['cv_generation']['duration']}s")
```

## Testing

### Unit Testing

```python
import pytest
from src.agents.content_writer import ContentWriterAgent

@pytest.mark.asyncio
async def test_content_generation():
    agent = ContentWriterAgent()
    result = await agent.generate_content(
        job_description="Test job",
        base_content="Test content"
    )
    assert result.content is not None
    assert len(result.content) > 0
```

### Integration Testing

```python
@pytest.mark.integration
async def test_full_workflow():
    orchestrator = EnhancedOrchestrator()
    result = await orchestrator.process_cv_request(
        job_description=SAMPLE_JOB_DESC,
        base_cv_content=SAMPLE_CV,
        session_id="test_session"
    )
    assert result.sections is not None
    assert len(result.big_10_skills) == 10
```

### E2E Testing

```python
@pytest.mark.e2e
async def test_complete_user_journey():
    # Test complete user workflow from input to output
    session_id = "e2e_test_session"
    
    # Step 1: Initialize session
    session_mgr.create_session(session_id)
    
    # Step 2: Process CV
    result = await orchestrator.process_cv_request(...)
    
    # Step 3: Verify results
    assert result.processing_time < 30.0
    assert all(section.items for section in result.sections)
```

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["python", "run_app.py"]
```

### Environment Setup

```bash
# Production environment
export GOOGLE_API_KEY=prod_api_key
export LOG_LEVEL=WARNING
export ENABLE_DEBUG_MODE=false

# Development environment
export GOOGLE_API_KEY=dev_api_key
export LOG_LEVEL=DEBUG
export ENABLE_DEBUG_MODE=true
```

## Security

### API Key Management

- Store API keys in environment variables
- Use secure key rotation practices
- Monitor API usage and rate limits

### Data Protection

- All PII is filtered from logs
- Session data is encrypted at rest
- Temporary files are automatically cleaned up

### Input Validation

```python
from src.utils.validation import validate_input

# All inputs are validated
validated_job_desc = validate_input(
    job_description,
    max_length=10000,
    required=True
)
```

For more detailed information, see the source code documentation and inline comments.