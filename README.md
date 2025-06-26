# AI CV Generator MVP

An intelligent, AI-powered CV tailoring application that creates personalized, job-specific CVs using advanced LLM technology and agentic workflows. Built with Python, Streamlit, and Google's Gemini AI.

## 🚀 Features

### Core Functionality
- **Intelligent CV Tailoring**: Advanced AI agents analyze job descriptions and automatically tailor CV content
- **Granular Item Control**: Accept, regenerate, or modify individual CV items (bullet points, qualifications)
- **"Big 10" Skills Extraction**: Automatically identifies and highlights the top 10 most relevant skills
- **Multi-Agent Architecture**: Specialized agents for content writing, research, QA, and formatting
- **Smart Fallbacks**: Robust error handling with graceful degradation when AI services are unavailable

### User Experience
- **Interactive Streamlit UI**: Modern, responsive interface for seamless CV creation
- **Real-time Processing**: Live feedback and progress tracking during CV generation
- **Session Persistence**: Save and resume work across sessions with automatic state management
- **Raw LLM Output Display**: View original AI responses for transparency and debugging
- **User Feedback Integration**: Provide feedback to improve AI-generated content

### Technical Excellence
- **LangGraph Orchestration**: Advanced workflow management with state persistence
- **Secure Logging**: Comprehensive logging with API key protection and PII filtering
- **Pydantic Data Models**: Type-safe data structures with validation
- **Comprehensive Testing**: Unit, integration, and E2E tests with 90%+ coverage
- **Performance Optimized**: CV generation typically completes in under 30 seconds

## Architecture Overview

The application follows a modular, service-oriented architecture with strict separation of concerns:

- **`instance/`**: All runtime-generated data is stored here, including logs, user sessions, vector databases, and output files. This directory is created automatically and is essential for application state. It should be excluded from version control and properly mounted in Docker deployments.
- **`src/`**: Contains the core application logic, organized by feature:
  - **`agents/`**: Specialized AI agents for content generation, analysis, and quality assurance.
  - **`api/`**: External API integrations (e.g., Google Gemini).
  - **`core/`**: Application startup, dependency injection, and core orchestration logic.
  - **`config/`**: Application settings, logging, and environment configuration.
  - **`error_handling/`**: Centralized error classes, boundaries, and agent-specific error handlers.
  - **`frontend/`**: Streamlit user interface components and callbacks.
  - **`integration/`**: High-level facade layer that unifies backend services and workflows.
  - **`models/`**: Pydantic data models with strict type validation and data contracts.
  - **`orchestration/`**: LangGraph workflow definitions and state management.
  - **`services/`**: Business logic for LLM interaction, session management, vector storage, etc.
  - **`templates/`**: Jinja2 templates for content generation and PDF rendering.
  - **`utils/`**: Shared utility functions including CV data manipulation factories.
- **`data/`**: Static configuration data, prompt templates, and persistent vector storage.
- **`tests/`**: Comprehensive testing suite with unit, integration, and end-to-end tests.
- **`docs/`**: Developer and user documentation including deployment guides.
- **`scripts/`**: Deployment and maintenance scripts.

## 🛠️ Getting Started

### Prerequisites

- **Python 3.11+**
- **Google Gemini API Key**
- **Git**

### Quick Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd aicvgen
   ```

2. **Create and activate a virtual environment:**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**

   Create a `.env` file in the project root and add your Google Gemini API key:

   ```.env
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```

5. **Run the application:**

   ```bash
   streamlit run app.py
   ```

6. **Access the application:**

   Open your browser to `http://localhost:8501`.

### Docker Installation (Alternative)

1. **Build the Docker image:**

   ```bash
   docker build -t aicvgen-app .
   ```

2. **Run the Docker container:**

   ```bash
   docker run -p 8501:8501 --env-file .env -v "%cd%/instance:/app/instance" aicvgen-app
   ```

## 📖 Usage Guide

### Basic Workflow

1. **Start a New Session**
   - Launch the application. A new session is created automatically.
   - Each session is automatically tracked and can be resumed

2. **Input Job Description**
   - Paste the target job description in the text area
   - The AI will automatically analyze requirements and skills

3. **Provide Base CV Content**
   - Upload an existing CV file (PDF, DOCX) or paste text directly
   - The system supports various CV formats and structures

4. **AI Processing**
   - Click "Generate Tailored CV" to start the AI workflow
   - Watch real-time progress as different agents process your content
   - Processing typically takes 15-30 seconds

5. **Review and Refine**
   - **Accept/Regenerate Items**: Click ✅ to accept or 🔄 to regenerate individual bullet points
   - **View Raw AI Output**: Toggle to see original LLM responses for transparency
   - **Provide Feedback**: Add specific feedback to improve regenerated content
   - **Big 10 Skills**: Review and modify the top 10 extracted skills

6. **Export Your CV**
   - Download the final CV in your preferred format
   - Save session state to resume later if needed

### Advanced Features

- **Session Management**: All work is automatically saved and can be resumed
- **Error Recovery**: The system gracefully handles API failures with fallback content
- **Performance Monitoring**: View processing times and system performance metrics
- **Debug Mode**: Enable detailed logging for troubleshooting

## 🔧 Development

### Architecture Overview

The AI CV Generator follows a modern, modular architecture with clear separation of concerns:

- **Multi-Agent System**: Specialized AI agents for different tasks (content writing, research, QA)
- **LangGraph Orchestration**: State-based workflow management with persistence
- **Pydantic Data Models**: Type-safe data structures with validation
- **Streamlit Frontend**: Interactive, real-time user interface
- **Secure Logging**: Comprehensive logging with PII protection

### Project Structure

```
aicvgen/
├── instance/              # Runtime data (logs, sessions, DBs) - gitignored
│   ├── logs/              # Application logs with structured format
│   ├── sessions/          # User session state persistence
│   ├── output/            # Generated CV files and documents
│   └── vector_db/         # ChromaDB persistent storage
├── src/
│   ├── agents/            # AI agent implementations
│   │   ├── agent_base.py      # Base agent class with common functionality
│   │   ├── cleaning_agent.py  # Content cleaning and normalization
│   │   ├── cv_analyzer_agent.py  # CV content analysis and optimization
│   │   ├── enhanced_content_writer.py  # Advanced content generation
│   │   ├── formatter_agent.py  # Document formatting and structure
│   │   ├── parser_agent.py     # CV and job description parsing
│   │   ├── quality_assurance_agent.py  # Content quality validation
│   │   ├── research_agent.py   # Job market research and analysis
│   │   └── specialized_agents.py  # Agent factory and specialized variants
│   ├── api/               # External API integrations
│   ├── config/            # Configuration management
│   │   ├── environment.py     # Environment variable handling
│   │   ├── logging_config.py  # Structured logging configuration
│   │   └── settings.py        # Application settings and validation
│   ├── core/              # Core application logic and orchestration
│   │   ├── application_startup.py  # Application initialization sequence
│   │   ├── dependency_injection.py  # DI container with lifecycle management
│   │   └── main.py             # Core application entry point
│   ├── error_handling/    # Centralized error management
│   │   ├── agent_error_handler.py  # Agent-specific error handling
│   │   ├── boundaries.py       # Error boundary definitions
│   │   ├── classification.py   # Error classification and routing
│   │   ├── exceptions.py       # Custom exception classes
│   │   └── models.py          # Error metadata models
│   ├── frontend/          # Streamlit UI components and callbacks
│   │   ├── callbacks.py       # UI event handlers and callbacks
│   │   └── ui_components.py   # Reusable UI components
│   ├── integration/       # High-level integration layer
│   │   └── enhanced_cv_system.py  # Unified CV generation facade
│   ├── models/            # Pydantic data models and schemas
│   │   ├── agent_models.py        # Agent execution models
│   │   ├── agent_output_models.py # Standardized agent output schemas
│   │   ├── data_models.py         # Core business data models
│   │   └── llm_service_models.py  # LLM service specific models
│   ├── orchestration/     # LangGraph workflow definitions
│   │   ├── cv_workflow_graph.py   # Main CV generation workflow
│   │   └── state.py              # Workflow state management
│   ├── services/          # Business logic services
│   │   ├── llm_service.py         # Enhanced LLM service with caching
│   │   ├── llm_cv_parser_service.py  # LLM-based parsing service
│   │   ├── session_manager.py     # User session management
│   │   └── vector_store_service.py  # ChromaDB integration
│   ├── templates/         # Jinja2 templates and content management
│   │   ├── content_templates.py   # Template management system
│   │   └── pdf_template.html      # PDF generation template
│   └── utils/             # Shared utility functions
│       ├── cv_data_factory.py     # CV data manipulation utilities
│       └── decorators.py          # Common decorators and helpers
├── data/                  # Static data and configurations
│   ├── prompts/           # LLM prompt templates
│   ├── templates/         # Document templates
│   └── vector_db/         # Vector database storage
├── tests/                 # Comprehensive testing suite
│   ├── unit/              # Unit tests for individual components
│   ├── integration/       # Integration tests for service interactions
│   └── e2e/              # End-to-end workflow tests
├── docs/                  # Documentation and development guides
│   ├── dev/               # Developer documentation
│   └── user/             # User guides and API reference
├── scripts/              # Deployment and maintenance scripts
│   ├── deploy.sh         # Production deployment script
│   └── migrate_logs.py   # Log migration utilities
├── logs/                 # Legacy log directory (deprecated)
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore patterns
├── app.py                # Main Streamlit application entry point
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Container deployment configuration
├── requirements.txt      # Python dependencies
├── pytest.ini           # Test configuration
└── README.md             # This documentation
```

### Development Setup

1. **Install development dependencies:**
```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

2. **Run tests:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test types
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest tests/e2e/          # E2E tests only
```

3. **Code quality checks:**
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Key Development Principles

- **Test-Driven Development**: Comprehensive test coverage with unit, integration, and E2E tests
- **Type Safety**: Full Pydantic model validation and mypy type checking
- **Secure by Design**: API key protection, PII filtering, and secure logging
- **Performance First**: Optimized for sub-30-second CV generation
- **Resilient Architecture**: Graceful error handling and fallback mechanisms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request
