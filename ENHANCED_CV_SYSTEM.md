# Enhanced CV Generation System

This document outlines the Enhanced CV Generation System, featuring advanced agents, intelligent orchestration, template management, and semantic vector database integration for superior CV creation and optimization.

## 🚀 New Features

### 1. Enhanced Agent System
- **Enhanced Content Writer Agent**: Improved content generation with better context awareness
- **CV Analysis Agent**: Analyzes CV content against job requirements
- **Content Optimization Agent**: Optimizes existing content for better impact
- **Quality Assurance Agent**: Performs comprehensive quality checks

### 2. Agent Orchestration
- **Multiple Orchestration Strategies**: Sequential, Parallel, Pipeline, and Adaptive execution
- **Task Dependencies**: Define complex workflows with task dependencies
- **Performance Monitoring**: Track agent performance and execution metrics
- **Error Recovery**: Automatic error handling and recovery mechanisms

### 3. Content Templates
- **Template Management**: Organized template system with categories
- **Dynamic Content**: Variable substitution and conditional content
- **Fallback Content**: Graceful degradation when templates are unavailable
- **Caching**: Efficient template caching for better performance

### 4. Vector Database Integration
- **Semantic Search**: Find similar content using embedding-based search
- **Content Storage**: Store and retrieve CV examples and templates
- **Similarity Matching**: Find relevant examples based on content similarity
- **Performance Optimization**: Efficient indexing and caching

### 5. Predefined Workflows
- **Basic CV Generation**: Essential sections with quality checks
- **Job-Tailored CV**: CV optimized for specific job requirements
- **CV Optimization**: Enhance existing CV content
- **Quality Assurance**: Comprehensive quality validation
- **Comprehensive CV**: Full-featured CV with all sections
- **Quick Update**: Fast updates for specific sections

### 6. Enhanced CV Generation API
- **RESTful Interface**: Complete API for all enhanced CV features
- **Request/Response Models**: Structured data models with validation
- **Error Handling**: Consistent error responses and logging
- **Performance Metrics**: Built-in performance tracking

## 📁 Project Structure

```
src/
├── agents/
│   ├── enhanced_content_writer.py    # Enhanced content writer agent
│   └── specialized_agents.py         # CV analysis, optimization, QA agents
├── templates/
│   └── content_templates.py          # Template management system
├── services/
│   └── vector_db.py                  # Enhanced vector database service
├── orchestration/
│   ├── agent_orchestrator.py         # Agent orchestration system
│   └── workflow_definitions.py       # Predefined workflow definitions
├── integration/
│   └── enhanced_cv_system.py          # Enhanced CV system integration layer
└── api/
    └── enhanced_cv_api.py             # Enhanced CV generation REST API
```

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8+
- Core system dependencies (from base implementation)
- Additional dependencies for enhanced features:

```bash
pip install faiss-cpu numpy scikit-learn
```

### Configuration

Create or update your configuration file:

```python
# config/settings.py
from src.integration.enhanced_cv_system import EnhancedCVConfig, IntegrationMode

ENHANCED_CV_CONFIG = EnhancedCVConfig(
    mode=IntegrationMode.PRODUCTION,
    enable_vector_db=True,
    enable_orchestration=True,
    enable_templates=True,
    enable_specialized_agents=True,
    vector_db_path="data/vector_db",
    template_cache_size=100,
    max_concurrent_agents=5,
    enable_performance_monitoring=True,
    enable_error_recovery=True
)
```

## 🚀 Quick Start

### Enhanced System Setup

### 1. Basic Usage

```python
from src.integration.enhanced_cv_system import get_enhanced_cv_integration
from src.orchestration.workflow_definitions import WorkflowType

# Get the integration instance
integration = get_enhanced_cv_integration()

# Generate a basic CV
result = await integration.generate_basic_cv(
    personal_info={
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "+1-555-0123"
    },
    experience=[
        {
            "title": "Software Engineer",
            "company": "Tech Corp",
            "start_date": "2020-01",
            "end_date": "2023-12",
            "description": "Developed web applications"
        }
    ],
    education=[
        {
            "degree": "Bachelor of Computer Science",
            "institution": "University of Technology",
            "graduation_date": "2020"
        }
    ]
)

print(f"Success: {result['success']}")
print(f"Generated content: {result['data']}")
```

### 2. Job-Tailored CV Generation

```python
# Generate a CV tailored to a specific job
result = await integration.generate_job_tailored_cv(
    personal_info={...},  # Same as above
    experience=[...],     # Same as above
    job_description={
        "title": "Senior Software Engineer",
        "company": "Innovation Inc",
        "description": "We're looking for a senior engineer...",
        "requirements": [
            "5+ years Python experience",
            "Experience with microservices",
            "Strong problem-solving skills"
        ]
    }
)
```

### 3. CV Optimization

```python
# Optimize an existing CV
result = await integration.optimize_cv(
    existing_cv={
        "personal_info": {...},
        "experience": [...],
        "education": [...]
    },
    target_role="Data Scientist",
    industry="Healthcare"
)
```

### 4. Content Search and Storage

```python
# Store content in vector database
document_id = await integration.store_content(
    content="Experienced software engineer with expertise in Python and machine learning",
    content_type=ContentType.EXPERIENCE,
    metadata={"industry": "technology", "level": "senior"}
)

# Search for similar content
results = await integration.search_content(
    query="Python machine learning experience",
    content_type=ContentType.EXPERIENCE,
    limit=5
)

for result in results:
    print(f"Content: {result['content']}")
    print(f"Similarity: {result['similarity']}")
```

### 5. Template Usage

```python
# List available templates
templates = integration.list_templates(category="experience")

# Format a template
formatted = integration.format_template(
    template_id="senior_engineer_experience",
    variables={
        "company": "Tech Corp",
        "years": "5",
        "technologies": "Python, React, AWS"
    }
)
```

## 🌐 REST API Usage

### Start the API Server

```python
# main.py
from fastapi import FastAPI
from src.api.enhanced_cv_api import router

app = FastAPI(title="Enhanced CV Generation System")
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### API Endpoints

#### CV Generation

```bash
# Generate basic CV
curl -X POST "http://localhost:8000/api/v2/cv/generate/basic" \
  -H "Content-Type: application/json" \
  -d '{
    "personal_info": {
      "name": "John Doe",
      "email": "john@example.com"
    },
    "experience": [...],
    "education": [...]
  }'

# Generate job-tailored CV
curl -X POST "http://localhost:8000/api/v2/cv/generate/job-tailored" \
  -H "Content-Type: application/json" \
  -d '{
    "personal_info": {...},
    "experience": [...],
    "education": [...],
    "job_description": {
      "title": "Senior Developer",
      "requirements": [...]
    }
  }'

# Optimize existing CV
curl -X POST "http://localhost:8000/api/v2/cv/optimize" \
  -H "Content-Type: application/json" \
  -d '{
    "existing_cv": {...},
    "target_role": "Data Scientist"
  }'
```

#### Content Management

```bash
# Search content
curl -X POST "http://localhost:8000/api/v2/content/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python machine learning",
    "content_type": "EXPERIENCE",
    "limit": 5
  }'

# Store content
curl -X POST "http://localhost:8000/api/v2/content/store" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Experienced Python developer...",
    "content_type": "EXPERIENCE",
    "metadata": {"industry": "tech"}
  }'
```

#### Template Management

```bash
# List templates
curl "http://localhost:8000/api/v2/templates?category=experience"

# Get specific template
curl "http://localhost:8000/api/v2/templates/senior_engineer_experience"

# Format template
curl -X POST "http://localhost:8000/api/v2/templates/format" \
  -H "Content-Type: application/json" \
  -d '{
    "template_id": "senior_engineer_experience",
    "variables": {
      "company": "Tech Corp",
      "years": "5"
    }
  }'
```

#### System Management

```bash
# Health check
curl "http://localhost:8000/api/v2/system/health"

# Performance statistics
curl "http://localhost:8000/api/v2/system/stats"

# List available workflows
curl "http://localhost:8000/api/v2/workflows"

# List available agents
curl "http://localhost:8000/api/v2/agents"
```

## 🔧 Advanced Configuration

### Custom Workflow Creation

```python
from src.orchestration.workflow_definitions import get_workflow_builder
from src.orchestration.agent_orchestrator import OrchestrationStrategy, AgentPriority
from datetime import timedelta

builder = get_workflow_builder()

# Create a custom workflow
custom_workflow = builder.create_custom_workflow(
    name="Custom CV Workflow",
    description="A custom workflow for specific requirements",
    workflow_type=WorkflowType.CUSTOM,
    agent_steps=[
        {
            "agent_type": "cv_analysis",
            "priority": AgentPriority.HIGH,
            "content_types": [ContentType.ANALYSIS],
            "timeout": timedelta(minutes=3),
            "dependencies": []
        },
        {
            "agent_type": "content_writer",
            "priority": AgentPriority.NORMAL,
            "content_types": [ContentType.EXPERIENCE],
            "timeout": timedelta(minutes=5),
            "dependencies": [0]
        }
    ],
    strategy=OrchestrationStrategy.SEQUENTIAL,
    required_inputs=["personal_info", "target_role"],
    optional_inputs=["preferences"]
)

# Execute the custom workflow
result = await builder.execute_workflow(
    workflow_type=custom_workflow.workflow_type,
    input_data={"personal_info": {...}, "target_role": "..."}
)
```

### Custom Agent Registration

```python
from src.agents.specialized_agents import AGENT_REGISTRY
from src.agents.agent_base import AgentBase

class CustomAgent(AgentBase):
    def __init__(self):
        super().__init__(
            name="Custom Agent",
            description="A custom agent for specific tasks"
        )

    async def run(self, context):
        # Custom agent logic
        return {"status": "success", "content": "Custom content"}

# Register the custom agent
AGENT_REGISTRY["custom"] = CustomAgent()
```

### Vector Database Customization

```python
from src.services.vector_db import VectorDB

# Custom embedding function
def custom_embedding_function(text):
    # Your custom embedding logic
    return embedding_vector

# Initialize with custom settings
vector_db = VectorDB(
    index_type="IndexIVFFlat",
    dimension=768,
    nlist=100,
    embed_function=custom_embedding_function
)
```

## 📊 Performance Monitoring

### Built-in Metrics

```python
# Get performance statistics
stats = integration.get_performance_stats()

print(f"Requests processed: {stats['requests_processed']}")
print(f"Average processing time: {stats['average_processing_time']:.2f}s")
print(f"Error rate: {stats['error_rate']:.2%}")
print(f"Cache hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']):.2%}")

# Component-specific stats
if 'vector_db' in stats:
    print(f"Vector DB documents: {stats['vector_db']['total_documents']}")
    print(f"Vector DB size: {stats['vector_db']['database_size_mb']:.2f} MB")

if 'orchestrator' in stats:
    print(f"Active tasks: {stats['orchestrator']['active_tasks']}")
    print(f"Completed tasks: {stats['orchestrator']['completed_tasks']}")
```

### Health Monitoring

```python
# System health check
health = integration.health_check()

print(f"Overall status: {health['status']}")
for component, status in health['components'].items():
    print(f"{component}: {status['status']}")
    if 'error' in status:
        print(f"  Error: {status['error']}")
```

## 🐛 Troubleshooting

### Common Issues

1. **Vector Database Initialization Fails**
   ```python
   # Check if FAISS is properly installed
   try:
       import faiss
       print("FAISS is available")
   except ImportError:
       print("Install FAISS: pip install faiss-cpu")
   ```

2. **Template Not Found**
   ```python
   # Check available templates
   templates = integration.list_templates()
   print(f"Available templates: {templates}")
   ```

3. **Agent Execution Timeout**
   ```python
   # Increase timeout in workflow configuration
   custom_options = {
       "timeout_multiplier": 2.0  # Double the default timeout
   }
   ```

4. **Memory Issues with Large Vector Database**
   ```python
   # Use IndexIVFFlat for large datasets
   config = EnhancedCVConfig(
       vector_db_config={
           "index_type": "IndexIVFFlat",
           "nlist": 1000  # Adjust based on dataset size
       }
   )
   ```

### Logging and Debugging

```python
# Enable debug mode
config = EnhancedCVConfig(
    debug_mode=True,
    enable_performance_monitoring=True
)

# Check logs
from src.core.logging_config import get_structured_logger
logger = get_structured_logger(__name__)
logger.info("Debug information", extra={"context": "debugging"})
```

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: Generate CVs in multiple languages
- **Industry-specific Templates**: Specialized templates for different industries
- **Real-time Collaboration**: Multiple users working on the same CV
- **Advanced Analytics**: Detailed analytics on CV performance
- **Integration APIs**: Connect with job boards and ATS systems

### Extension Points
- **Custom Agents**: Easy framework for adding new agent types
- **Plugin System**: Modular architecture for extending functionality
- **Custom Workflows**: Visual workflow builder
- **External Integrations**: Connect with external services and APIs

## 📝 Contributing

### Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd aicvgen

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

### Adding New Features

1. **New Agent Types**: Extend `AgentBase` and register in `AGENT_REGISTRY`
2. **New Workflows**: Use `WorkflowBuilder.create_custom_workflow()`
3. **New Templates**: Add to `ContentTemplateManager`
4. **New API Endpoints**: Add to `enhanced_cv_api.py`

### Testing

```python
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Performance tests
python -m pytest tests/performance/
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🤝 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting section

---

**Enhanced CV System Implementation Complete** ✅

The Enhanced CV Generation System provides a comprehensive, scalable, and extensible solution with advanced AI-powered features for professional CV creation. The modular architecture allows for easy customization and extension while maintaining high performance and reliability.