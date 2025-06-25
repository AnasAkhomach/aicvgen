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

The application follows a modular, service-oriented architecture:

- **`instance/`**: All runtime-generated data is stored here, including logs, user sessions, vector databases, and output files. This directory is created automatically and is essential for application state. It should be excluded from version control.
- **`src/`**: Contains the core application logic, organized by feature:
  - **`api/`**: External API integrations (e.g., Google Gemini).
  - **`agents/`**: Specialized AI agents for content generation and analysis.
  - **`core/`**: Application startup, dependency injection, and core orchestration logic.
  - **`config/`**: Application settings, logging, and environment configuration.
  - **`error_handling/`**: Centralized error classes and utilities.
  - **`frontend/`**: Streamlit user interface components and callbacks.
  - **`services/`**: Business logic for session management, vector storage, etc.
  - **`utils/`**: Shared utility functions.
- **`tests/`**: Contains all unit, integration, and end-to-end tests.
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
   GOOGLE_API_KEY="your_gemini_api_key_here"
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
   docker build -t aicvgen .
   ```

2. **Run the container:**

   The `instance` directory is mounted as a volume to persist application data across container restarts.

   ```bash
   docker run -p 8501:8501 --name aicvgen-app -v "%cd%/instance:/app/instance" --env-file .env aicvgen
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
├── src/
│   ├── agents/              # AI agent implementations
│   │   ├── base_agent.py    # Base agent class with common functionality
│   │   ├── content_writer.py # Content generation and tailoring
│   │   ├── research_agent.py # Job analysis and skill extraction
│   │   ├── qa_agent.py      # Quality assurance and validation
│   │   └── formatter.py     # CV formatting and structure
│   ├── core/                # Core business logic
│   │   ├── orchestrator.py  # Main workflow orchestration
│   │   ├── session_manager.py # Session state management
│   │   ├── state_manager.py # Application state persistence
│   │   └── workflow.py      # LangGraph workflow definitions
│   ├── data/                # Data models and schemas
│   │   ├── models.py        # Pydantic data models
│   │   ├── schemas.py       # API and validation schemas
│   │   └── enums.py         # Enumeration definitions
│   ├── services/            # External service integrations
│   │   ├── llm_service.py   # Google Gemini LLM integration
│   │   ├── file_service.py  # File I/O operations
│   │   └── export_service.py # CV export functionality
│   ├── ui/                  # User interface components
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/          # Streamlit page definitions
│   │   └── utils.py        # UI utility functions
│   └── utils/              # Utility functions
│       ├── logging.py      # Secure logging utilities
│       ├── validation.py   # Data validation helpers
│       └── helpers.py      # General utility functions
├── tests/
│   ├── unit/               # Unit tests (90%+ coverage)
│   │   ├── test_agents/    # Agent-specific tests
│   │   ├── test_core/      # Core logic tests
│   │   ├── test_services/  # Service integration tests
│   │   └── test_utils/     # Utility function tests
│   ├── integration/        # Integration tests
│   │   ├── test_workflows/ # End-to-end workflow tests
│   │   └── test_api/       # API integration tests
│   ├── e2e/               # End-to-end tests
│   │   ├── test_complete_cv_generation.py # Full workflow tests
│   │   ├── test_individual_item_processing.py # Granular processing
│   │   ├── test_error_recovery.py # Error handling and resilience
│   │   ├── conftest.py     # Test configuration and fixtures
│   │   └── test_data/      # Test data and mock responses
│   └── conftest.py         # Global test configuration
├── data/
│   ├── input/              # Sample input files and templates
│   ├── output/             # Generated CV outputs
│   ├── sessions/           # Session state storage
│   └── templates/          # CV templates and formats
├── config/                 # Configuration files
│   ├── logging.yaml        # Logging configuration
│   └── app_config.yaml     # Application settings
├── docs/
│   ├── dev/               # Development documentation
│   └── user/              # User documentation
├── logs/                   # Application logs (auto-created)
├── scripts/               # Utility and deployment scripts
├── .vs_venv/              # Virtual environment (local)
├── app.py                 # Main Streamlit application
├── run_app.py             # Application launcher with environment setup
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container deployment configuration
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore patterns
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
