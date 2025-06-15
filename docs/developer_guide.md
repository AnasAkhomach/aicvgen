# AI CV Generator - Developer Guide

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
5. [Agent System](#agent-system)
6. [API Reference](#api-reference)
7. [Testing](#testing)
8. [Performance Optimization](#performance-optimization)
9. [Deployment](#deployment)
10. [Contributing](#contributing)
11. [Troubleshooting](#troubleshooting)

## Architecture Overview

The AI CV Generator is built using a modern, modular architecture with the following key principles:

- **Agent-Based Processing**: Specialized AI agents handle different aspects of CV generation
- **Async Processing**: Non-blocking operations for better performance
- **State Management**: Centralized state tracking throughout the workflow
- **Error Recovery**: Robust error handling with automatic retry mechanisms
- **Caching**: Intelligent caching for improved performance
- **Monitoring**: Comprehensive performance and error monitoring

### Technology Stack
- **Backend**: Python 3.9+
- **Web Framework**: Streamlit
- **AI/LLM**: Google Gemini API
- **Workflow**: LangGraph
- **Database**: SQLAlchemy with SQLite
- **Testing**: pytest
- **Containerization**: Docker
- **Monitoring**: Custom performance monitoring

## Development Setup

### Prerequisites
- Python 3.9 or higher
- Git
- Docker (optional, for containerized development)
- Google Gemini API key

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd aicvgen
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .vs_venv
   # Windows
   .vs_venv\Scripts\activate
   # Linux/Mac
   source .vs_venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py --server.port=8501
   ```

### Docker Development

1. **Build and Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Development with Hot Reload**
   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

## Project Structure

```
aicvgen/
├── src/                          # Source code
│   ├── agents/                   # AI agents
│   │   ├── base_agent.py        # Base agent class
│   │   ├── parser_agent.py      # CV/JD parsing
│   │   ├── research_agent.py    # Job research
│   │   ├── content_writer.py    # Content generation
│   │   ├── quality_agent.py     # Quality assurance
│   │   └── formatter_agent.py   # Output formatting
│   ├── core/                    # Core functionality
│   │   ├── state_manager.py     # State management
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── workflow.py          # LangGraph workflow
│   ├── services/                # External services
│   │   ├── llm.py              # LLM service
│   │   ├── file_service.py     # File operations
│   │   └── vector_service.py   # Vector database
│   ├── utils/                   # Utilities
│   │   ├── performance.py      # Performance monitoring
│   │   ├── logging_config.py   # Logging setup
│   │   └── rate_limiter.py     # Rate limiting
│   └── frontend/               # Streamlit UI
│       └── streamlit_app.py    # Main UI
├── tests/                       # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── docs/                       # Documentation
├── data/                       # Data files
├── logs/                       # Log files
└── scripts/                    # Utility scripts
```

## Core Components

### State Manager
Centralized state management for the entire workflow.

```python
from src.core.state_manager import StateManager, WorkflowState

# Initialize state
state_manager = StateManager()
state = WorkflowState(
    job_description="...",
    cv_content="...",
    user_preferences={"format": "pdf"}
)

# Update state
state_manager.update_state(state, "processing_stage", "parsing")
```

### LLM Service
Manages interactions with the Gemini API with caching and error recovery.

```python
from src.services.llm import get_llm_service

llm_service = get_llm_service()
response = await llm_service.generate_content(
    prompt="Analyze this CV...",
    context={"cv_content": cv_text}
)
```

### Error Handling
Robust error handling with custom exceptions and retry logic.

```python
from src.core.exceptions import (
    CVGenerationError,
    LLMServiceError,
    ValidationError
)
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
async def process_with_retry():
    try:
        result = await some_operation()
        return result
    except LLMServiceError as e:
        logger.error(f"LLM service error: {e}")
        raise
```

## Agent System

The application uses specialized agents for different processing stages:

### Base Agent
All agents inherit from `BaseAgent` which provides common functionality:

```python
from src.agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    async def process(self, state: WorkflowState) -> WorkflowState:
        # Agent-specific processing logic
        result = await self.llm_service.generate_content(
            prompt=self.get_prompt(state),
            context=state.context
        )
        
        # Update state with results
        state.results["custom_analysis"] = result.content
        return state
```

### Parser Agent
Extracts and structures content from CVs and job descriptions:

```python
from src.agents.parser_agent import ParserAgent

parser = ParserAgent()
state = await parser.process(state)
# state.parsed_cv and state.parsed_job_description are now populated
```

### Research Agent
Researches job requirements and industry standards:

```python
from src.agents.research_agent import ResearchAgent

researcher = ResearchAgent()
state = await researcher.process(state)
# state.research_results contains job analysis
```

### Content Writer Agent
Generates enhanced CV content:

```python
from src.agents.enhanced_content_writer import EnhancedContentWriter

writer = EnhancedContentWriter()
state = await writer.process(state)
# state.enhanced_content contains improved CV content
```

### Quality Assurance Agent
Validates and improves content quality:

```python
from src.agents.quality_assurance_agent import QualityAssuranceAgent

qa_agent = QualityAssuranceAgent()
state = await qa_agent.process(state)
# state.quality_report contains validation results
```

### Formatter Agent
Generates final output in requested formats:

```python
from src.agents.formatter_agent import FormatterAgent

formatter = FormatterAgent()
state = await formatter.process(state)
# state.formatted_outputs contains final files
```

## API Reference

### Workflow Execution

```python
from src.core.workflow import CVGenerationWorkflow

# Initialize workflow
workflow = CVGenerationWorkflow()

# Execute workflow
result = await workflow.execute(
    cv_file=cv_file,
    job_description=job_description,
    preferences=user_preferences
)
```

### File Service

```python
from src.services.file_service import FileService

file_service = FileService()

# Extract text from file
text = await file_service.extract_text(file_path)

# Generate PDF
pdf_path = await file_service.generate_pdf(content, output_path)

# Generate DOCX
docx_path = await file_service.generate_docx(content, output_path)
```

### Vector Service

```python
from src.services.vector_service import VectorService

vector_service = VectorService()

# Store embeddings
await vector_service.store_embedding(text, metadata)

# Search similar content
results = await vector_service.search_similar(query, top_k=5)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_parser_agent.py -v
```

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete workflows

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from src.agents.parser_agent import ParserAgent

class TestParserAgent:
    @pytest.fixture
    def parser_agent(self):
        return ParserAgent()
    
    @pytest.fixture
    def mock_llm_service(self):
        mock = Mock()
        mock.generate_content.return_value = Mock(
            content="{\"skills\": [\"Python\", \"AI\"]}",
            success=True
        )
        return mock
    
    @pytest.mark.asyncio
    async def test_parse_cv(self, parser_agent, mock_llm_service):
        with patch.object(parser_agent, 'llm_service', mock_llm_service):
            state = WorkflowState(cv_content="Sample CV content")
            result = await parser_agent.process(state)
            
            assert result.parsed_cv is not None
            assert "skills" in result.parsed_cv
```

## Performance Optimization

### Caching
The application implements multi-level caching:

```python
from src.utils.performance import monitor_performance

@monitor_performance("expensive_operation")
async def expensive_operation(data):
    # This operation will be monitored
    result = await process_data(data)
    return result
```

### Memory Management

```python
from src.utils.performance import auto_memory_optimize

@auto_memory_optimize(threshold_mb=100)
def memory_intensive_function():
    # Automatic memory optimization when threshold exceeded
    pass
```

### Batch Processing

```python
from src.utils.performance import get_batch_processor

batch_processor = get_batch_processor()
results = await batch_processor.process_batch_async(
    items=data_items,
    processor=process_item,
    progress_callback=update_progress
)
```

### Performance Monitoring

```python
from src.utils.performance import get_performance_monitor

monitor = get_performance_monitor()

# Get operation statistics
stats = monitor.get_operation_stats("llm_call")

# Export metrics
monitor.export_metrics("performance_report.json")
```

## Deployment

### Environment Variables

```bash
# Required
GEMINI_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key

# Optional
LOG_LEVEL=INFO
CACHE_TTL=3600
MAX_CONCURRENT_REQUESTS=10
DATABASE_URL=sqlite:///data/app.db
```

### Docker Deployment

```bash
# Build image
docker build -t aicvgen:latest .

# Run container
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key \
  -v $(pwd)/data:/app/data \
  aicvgen:latest
```

### Production Considerations

1. **Security**:
   - Use environment variables for secrets
   - Implement proper authentication
   - Enable HTTPS
   - Regular security updates

2. **Scalability**:
   - Use load balancers for multiple instances
   - Implement proper caching strategies
   - Monitor resource usage
   - Set up auto-scaling

3. **Monitoring**:
   - Set up logging aggregation
   - Implement health checks
   - Monitor performance metrics
   - Set up alerting

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Maintain test coverage above 80%

### Development Workflow
1. Create feature branch from main
2. Implement changes with tests
3. Run full test suite
4. Update documentation
5. Submit pull request

### Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No security vulnerabilities
- [ ] Performance impact considered
- [ ] Error handling is robust

## Troubleshooting

### Common Development Issues

**Import Errors**
```bash
# Ensure PYTHONPATH includes src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**LLM Service Issues**
```python
# Check API key configuration
from src.services.llm import get_llm_service
service = get_llm_service()
stats = service.get_service_stats()
print(f"API calls: {stats['total_calls']}")
```

**Performance Issues**
```python
# Monitor performance
from src.utils.performance import get_performance_monitor
monitor = get_performance_monitor()
stats = monitor.get_overall_stats()
print(f"Average duration: {stats['average_duration_seconds']}s")
```

**Memory Issues**
```python
# Check memory usage
from src.utils.performance import get_memory_optimizer
optimizer = get_memory_optimizer()
memory_info = optimizer.get_memory_info()
print(f"Memory usage: {memory_info['rss_mb']}MB")
```

### Debugging Tips

1. **Enable Debug Logging**:
   ```python
   import logging
   logging.getLogger().setLevel(logging.DEBUG)
   ```

2. **Use Performance Monitoring**:
   ```python
   from src.utils.performance import monitor_performance
   
   @monitor_performance("debug_operation")
   def debug_function():
       # Your code here
       pass
   ```

3. **Check State Management**:
   ```python
   # Log state transitions
   state_manager.enable_debug_logging()
   ```

4. **Monitor LLM Calls**:
   ```python
   # Enable LLM call logging
   llm_service.enable_debug_mode()
   ```

### Getting Help

- Check the logs in `logs/` directory
- Review error messages carefully
- Use the debugging utilities provided
- Consult the architecture documentation
- Check existing issues and tests for examples

---

*This developer guide is maintained alongside the codebase. Please update it when making significant changes to the architecture or adding new features.*