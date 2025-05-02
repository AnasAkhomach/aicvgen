# CV Tailoring AI Agent - MVP Upgrade

## Overview
This project is an AI-powered CV tailoring system that adapts a user's CV to match specific job descriptions. The system parses both the job description and user's existing CV, and then generates tailored content that highlights relevant skills and experience.

## Recent Upgrades
The codebase has been significantly enhanced with a more structured data model that enables granular content management:

1. **StructuredCV Data Model**
   - Implemented a hierarchical structure (Section > Subsection > Item)
   - Added status tracking via enums (ItemStatus) for workflow management
   - Created methods for content manipulation and persistence

2. **Enhanced Parser Agent**
   - Improved CV parsing to extract structured content from Markdown
   - Added support for a "Start from Scratch" option
   - Implemented detailed metadata extraction

3. **Upgraded Content Writer Agent**
   - Updated to work with the new data model
   - Added support for granular content generation
   - Implemented tailored prompt construction for different section types
   - Enhanced job focus extraction with better identification of key skills and responsibilities
   - Improved handling of content generation with customized prompting per section type

4. **Formatter Agent Improvements**
   - Fixed handling of ContentData structure
   - Added intelligent completion of truncated bullet points
   - Improved formatting consistency across different section types
   - Enhanced language section formatting
   - Added better fallback formatting for error cases

5. **Session Persistence**
   - Added JSON serialization for complete CV data
   - Created a data/sessions directory structure for storage
   - Implemented session loading/saving functionality

6. **UI Improvements**
   - Updated the interface to support card-based content review
   - Added controls for accepting, editing, and regenerating specific content items
   - Implemented user feedback collection at the item level

7. **Better Error Handling**
   - Added robust debugging and error logging throughout the pipeline
   - Implemented proper fallback mechanisms in case of failures
   - Fixed truncated content through post-processing

## Bug Fixes
- Fixed linter errors in parser_agent.py (removed invalid 'raw_text' parameter)
- Updated Streamlit code to use the appropriate rerun method
- Fixed duplicate skills in the key qualifications section
- Resolved issues with truncated bullet points in project descriptions
- Fixed language section formatting problems

## Next Steps

### Immediate Tasks
1. **Testing System Improvement**
   - Develop more comprehensive test cases
   - Implement automated regression testing
   - Add more detailed logging for debugging

2. **QA Agent Upgrades**
   - Enhance quality checks for generated content
   - Implement feedback-driven content improvement

3. **Research Agent Integration**
   - Further integrate with company research capabilities
   - Improve industry-specific term detection

### Medium-Term Goals
1. **Formatter Improvements**
   - Add more export formats (PDF, DOCX, HTML)
   - Implement custom styling options

2. **UI Refinements**
   - Add progress tracking
   - Improve content editing interface
   - Add visual differentiation between original and AI-generated content

3. **Comprehensive Testing**
   - Add unit tests for all components
   - Implement integration tests
   - Create sample CV and job description datasets for validation

### Long-Term Vision
1. **Learning System**
   - Train on user feedback to improve content generation
   - Build a recommendation system for skills highlighting

2. **Multi-language Support**
   - Add support for CVs in different languages
   - Implement translation capabilities

3. **Integration Capabilities**
   - Add API for integration with job boards
   - Create plugins for popular word processors

## Installation and Setup
```bash
# Clone repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run main.py
```

## Usage
1. Enter a job description or paste a job listing
2. Upload your existing CV or start from scratch
3. Review and edit AI-generated content
4. Export the final tailored CV in your preferred format

## Project Structure
- `state_manager.py`: Core data model and state management
- `parser_agent.py`: Job description and CV parsing
- `content_writer_agent.py`: Content generation and tailoring
- `formatter_agent.py`: Formatting and styling of CV content
- `main.py`: Streamlit UI and application logic 