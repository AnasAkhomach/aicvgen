# End-to-End (E2E) Tests for CV Tailoring Application

This directory contains comprehensive end-to-end tests for the CV tailoring application, implementing the testing strategy outlined in the `testing_action_plan.txt`.

## Overview

The E2E tests validate three critical user workflows:

1. **Complete CV Generation** - Full workflow through all agents
2. **Individual Item Processing** - Role-by-role generation with rate limiting
3. **Error Recovery** - Graceful handling of failures and edge cases

## Directory Structure

```
tests/e2e/
├── __init__.py                     # E2E test package initialization
├── conftest.py                     # Shared fixtures and configuration
├── run_e2e_tests.py               # Test runner script
├── README.md                       # This documentation
├── test_complete_cv_generation.py  # Complete CV workflow tests
├── test_individual_item_processing.py # Individual item processing tests
├── test_error_recovery.py          # Error handling and recovery tests
└── test_data/                      # Test data and mock responses
    ├── __init__.py
    ├── sample_job_descriptions.py   # Sample job descriptions for testing
    ├── sample_base_cvs.py          # Sample base CVs with different formats
    ├── mock_responses.py           # Mock LLM responses and API errors
    └── expected_outputs.py         # Expected results for validation
```

## Test Data Strategy

### Sample Data Sets

- **Job Descriptions**: Tech roles (AI Engineer, Data Scientist, Software Developer)
- **Base CVs**: Various experience levels (Junior, Mid-level, Senior) and formats
- **Expected Outputs**: Pre-validated CV sections for comparison

### Mock Responses

- **LLM Outputs**: Realistic "Big 10" skills, experience bullets, professional summaries
- **API Errors**: Rate limit, timeout, authentication failures
- **Vector Search**: Relevant CV content matches

## Running E2E Tests

### Quick Start

```bash
# Run all E2E tests
python tests/e2e/run_e2e_tests.py

# Run specific test suite
python tests/e2e/run_e2e_tests.py --suite complete_cv

# Run with verbose output
python tests/e2e/run_e2e_tests.py --verbose

# Run for specific job roles
python tests/e2e/run_e2e_tests.py --roles software_engineer ai_engineer
```

### Using pytest directly

```bash
# Run all E2E tests
pytest tests/e2e/ -m e2e -v

# Run specific test file
pytest tests/e2e/test_complete_cv_generation.py -v

# Run with coverage
pytest tests/e2e/ -m e2e --cov=src --cov-report=html

# Run in parallel
pytest tests/e2e/ -m e2e -n 4
```

### Advanced Options

```bash
# Use real LLM (requires API keys)
python tests/e2e/run_e2e_tests.py --real-llm

# Enable performance profiling
python tests/e2e/run_e2e_tests.py --profile

# Custom output directory
python tests/e2e/run_e2e_tests.py --output-dir ./my_results

# Parallel execution
python tests/e2e/run_e2e_tests.py --workers 4

# Custom timeout
python tests/e2e/run_e2e_tests.py --timeout 900
```

## Test Suites

### 1. Complete CV Generation (`test_complete_cv_generation.py`)

**Purpose**: Validates the full CV tailoring workflow from job description to final PDF.

**Test Cases**:
- `test_complete_cv_generation_workflow` - End-to-end CV generation
- `test_cv_content_quality_validation` - Content quality and structure
- `test_cv_formatting_and_structure` - PDF formatting and layout
- `test_processing_time_performance` - Performance benchmarks
- `test_ats_compatibility` - ATS optimization validation

**Assertions**:
- PDF structure and content quality
- "Big 10" skills integration
- Experience bullets enhancement
- Processing time within thresholds
- ATS compatibility scores

### 2. Individual Item Processing (`test_individual_item_processing.py`)

**Purpose**: Tests role-by-role processing with rate limiting and user feedback.

**Test Cases**:
- `test_individual_experience_processing` - Single role processing
- `test_rate_limit_compliance` - Rate limiting behavior
- `test_user_feedback_integration` - Feedback incorporation
- `test_parallel_item_processing` - Concurrent processing
- `test_processing_quality_consistency` - Quality across items

**Assertions**:
- Rate limit compliance
- User feedback integration
- Processing consistency
- Performance optimization

### 3. Error Recovery (`test_error_recovery.py`)

**Purpose**: Validates error handling, recovery mechanisms, and graceful degradation.

**Test Cases**:
- `test_invalid_job_description_handling` - Invalid input handling
- `test_api_failure_recovery` - API error recovery
- `test_rate_limit_error_handling` - Rate limit management
- `test_timeout_error_recovery` - Timeout handling
- `test_graceful_degradation` - Fallback mechanisms
- `test_state_preservation_during_errors` - State consistency

**Assertions**:
- Graceful error handling
- User notification systems
- State preservation
- Recovery success rates

## Configuration

### Test Configuration (`conftest.py`)

The `conftest.py` file provides:

- **Fixtures**: Shared test components (orchestrator, mock services, test data)
- **Configuration**: Test timeouts, performance thresholds, quality criteria
- **Utilities**: Validation helpers, performance timers, error simulators

### Environment Variables

```bash
# Optional: Real LLM API configuration
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"

# Test configuration
export E2E_TEST_TIMEOUT=600
export E2E_PARALLEL_WORKERS=2
export E2E_OUTPUT_DIR="./test_results"
```

## Test Data

### Job Descriptions (`test_data/sample_job_descriptions.py`)

Provides realistic job descriptions for:
- Software Engineer
- AI Engineer  
- Data Scientist

Each includes:
- Role requirements
- Technical skills
- Experience expectations
- Company context

### Base CVs (`test_data/sample_base_cvs.py`)

Sample CVs with different:
- Experience levels (Junior, Mid-level, Senior)
- Formatting styles
- Content structures
- Technical backgrounds

### Expected Outputs (`test_data/expected_outputs.py`)

Pre-validated CV sections including:
- Professional summaries
- Experience bullets
- Technical skills
- Project descriptions

With quality criteria:
- Content requirements
- Formatting standards
- ATS compatibility
- Performance metrics

## Validation and Quality Metrics

### Content Quality

- **Readability**: Flesch reading ease, sentence length
- **Professional Language**: Action verbs, passive voice limits
- **Quantification**: Specific numbers, percentage improvements
- **Technical Relevance**: Keyword matching, skill coverage

### Formatting Standards

- **Structure**: Consistent headers, logical order
- **Typography**: Font usage, spacing, hierarchy
- **Organization**: Chronological order, content grouping
- **Length**: Page limits, section balance

### ATS Compatibility

- **Parsing**: Standard headers, simple formatting
- **Keywords**: Job title variations, skill density
- **File Format**: PDF compatibility, text extraction

## Performance Benchmarks

### Processing Time Thresholds

- **Complete CV Generation**: < 3 minutes
- **Individual Item Processing**: < 1 minute
- **Error Recovery**: < 2 minutes

### Quality Thresholds

- **Content Score**: ≥ 80%
- **Formatting Score**: ≥ 90%
- **ATS Compatibility**: ≥ 85%
- **Success Rate**: ≥ 95%

## Reporting

### Generated Reports

1. **Console Summary**: Real-time test progress and results
2. **JSON Report**: Machine-readable detailed results
3. **HTML Report**: Interactive test report with charts
4. **Markdown Report**: Human-readable summary
5. **JUnit XML**: CI/CD integration format
6. **Coverage Report**: Code coverage analysis (if enabled)

### Report Contents

- Test execution summary
- Individual test results
- Performance metrics
- Quality validation results
- Error analysis
- Recommendations

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: E2E Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  e2e-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-html pytest-cov pytest-xdist
    
    - name: Run E2E Tests
      run: |
        python tests/e2e/run_e2e_tests.py --suite all --workers 2 --profile
    
    - name: Upload Test Results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: e2e-test-results
        path: test_results/
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure project root is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Timeout Errors**
   ```bash
   # Increase timeout for slow environments
   python tests/e2e/run_e2e_tests.py --timeout 1200
   ```

3. **Mock LLM Issues**
   ```bash
   # Verify mock responses are properly configured
   pytest tests/e2e/test_data/ -v
   ```

4. **Permission Errors**
   ```bash
   # Ensure output directory is writable
   mkdir -p test_results
   chmod 755 test_results
   ```

### Debug Mode

```bash
# Run with maximum verbosity and debugging
python tests/e2e/run_e2e_tests.py --verbose --suite complete_cv --workers 1

# Run single test with pdb
pytest tests/e2e/test_complete_cv_generation.py::test_complete_cv_generation_workflow -v -s --pdb
```

## Contributing

### Adding New Tests

1. **Create test file** in appropriate test suite
2. **Add test data** to `test_data/` directory
3. **Update expected outputs** if needed
4. **Add fixtures** to `conftest.py` if required
5. **Update documentation** in this README

### Test Naming Conventions

- Test files: `test_<feature>_<workflow>.py`
- Test functions: `test_<specific_scenario>`
- Fixtures: `<component>_<type>` (e.g., `mock_llm_client`)
- Test data: `SAMPLE_<DATA_TYPE>` (e.g., `SAMPLE_JOB_DESCRIPTIONS`)

### Quality Guidelines

- **Isolation**: Tests should be independent
- **Repeatability**: Consistent results across runs
- **Performance**: Reasonable execution time
- **Coverage**: Comprehensive scenario coverage
- **Documentation**: Clear test purpose and assertions

## Future Enhancements

### Planned Features

1. **Visual Regression Testing**: PDF layout comparison
2. **Load Testing**: High-volume processing scenarios
3. **Security Testing**: Input validation and sanitization
4. **Accessibility Testing**: PDF accessibility compliance
5. **Multi-language Support**: International CV formats

### Integration Opportunities

1. **Real LLM Testing**: Periodic validation with actual APIs
2. **User Acceptance Testing**: Integration with user feedback
3. **Performance Monitoring**: Continuous performance tracking
4. **Quality Metrics**: Automated quality assessment

---

## Support

For questions or issues with E2E testing:

1. Check this documentation
2. Review test logs in output directory
3. Run tests with `--verbose` flag
4. Check project documentation in `docs/`
5. Create issue with test results and logs