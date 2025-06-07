# AI CV Generator

This project is an AI-powered CV tailoring tool that helps users create personalized CVs for specific job descriptions.

## Features

- **AI-driven CV tailoring**: Automatically tailors your CV to match job descriptions
- **Section-level control**: Edit and regenerate content at the section level for a simpler user experience
- **Interactive UI**: Review and edit AI-generated content before finalizing
- **Template-based generation**: Creates professionally formatted CVs
- **Session management**: Save and resume your work

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt

### Installation

1. Clone the repository:
```bash
git clone [repository URL]
cd aicvgen # Or your repository directory name
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
   - For the Streamlit UI:
     ```bash
     streamlit run run_app.py
     ```
   - For the FastAPI backend (if applicable):
     ```bash
     python src/api/main.py
     ```

## Usage

1. Enter your job description
2. Upload your existing CV or start from scratch
3. Review and edit the AI-generated content by section
4. Accept or regenerate sections as needed
5. Export your tailored CV

## Development

### Project Structure

The project is organized as follows:

- `run_app.py`: Main entry point for the Streamlit application.
- `src/`: Contains the core source code.
  - `core/`: Core logic of the application.
    - `main.py`: Core application logic for Streamlit.
    - `orchestrator.py`: Manages the overall workflow.
    - `state_manager.py`: Manages application state and data structures.
  - `agents/`: Contains various AI agents.
    - `content_writer_agent.py`: Handles AI content generation.
    - `parser_agent.py`: Parses job descriptions and CVs.
    - `cv_analyzer_agent.py`: Analyzes CV content.
    - `formatter_agent.py`: Formats the output CV.
    - `quality_assurance_agent.py`: Ensures quality of generated content.
    - `research_agent.py`: Performs research tasks.
    - `tools_agent.py`: Provides tools for other agents.
    - `vector_store_agent.py`: Manages vector store interactions.
  - `api/`: Contains the FastAPI backend code.
    - `main.py`: Entry point for the FastAPI application.
  - `config/`: Configuration files (e.g., logging).
  - `frontend/`: Files related to the user interface (HTML, CSS, JS).
  - `models/`: Data models and structures.
  - `services/`: External services integrations (e.g., LLM, Vector DB).
  - `templates/`: CV templates and other Jinja2 templates.
  - `utils/`: Utility functions and classes.
- `data/`: Stores data used by the application, including job descriptions, prompts, and user sessions.
- `docs/`: Contains documentation files (SRS, SDD).
- `logs/`: Application logs.
- `scripts/`: Utility scripts for development and maintenance.
- `tests/`: Contains unit and integration tests.
  - `unit/`: Unit tests for individual modules.
  - `integration/`: Integration tests for component interactions.
- `requirements.txt`: Lists Python dependencies.
- `Dockerfile`: For building the application container.
- `README.md`: This file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request 
