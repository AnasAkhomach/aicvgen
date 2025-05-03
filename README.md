# AI CV Generator

This project is an AI-powered CV tailoring tool that helps users create personalized CVs for specific job descriptions.

## Features

- **AI-driven CV tailoring**: Automatically tailors your CV to match job descriptions
- **Section-level control**: Edit and regenerate content at the section level for a simpler user experience
- **Interactive UI**: Review and edit AI-generated content before finalizing
- **Template-based generation**: Creates professionally formatted CVs
- **Session management**: Save and resume your work

## Recent Changes

### v1.3: Section-Level Control

The system has been updated to use section-level control rather than granular item-level control. This simplifies the user interface and workflow while maintaining powerful tailoring capabilities.

Key improvements:
- Content generation and regeneration now works at the section level
- UI controls (Accept, Regenerate) operate on entire sections
- The orchestrator and content writer agents have been updated to support section-level operations
- Removed unused log files and fixed code quality issues

These changes make the application more user-friendly while maintaining all the core functionality.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in requirements.txt

### Installation

1. Clone the repository:
```bash
git clone [repository URL]
cd aicvgen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Enter your job description
2. Upload your existing CV or start from scratch
3. Review and edit the AI-generated content by section
4. Accept or regenerate sections as needed
5. Export your tailored CV

## Development

### Project Structure

- `main.py`: Streamlit application entry point
- `orchestrator.py`: Manages the overall workflow
- `state_manager.py`: Manages application state and data structures
- `content_writer_agent.py`: Handles AI content generation
- `parser_agent.py`: Parses job descriptions and CVs
- Other agents: Research, formatting, quality assurance, etc.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request 