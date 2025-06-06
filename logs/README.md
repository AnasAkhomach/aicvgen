# Logs Directory

This directory contains application logs for the CV AI Generator application.

## Log Structure

### Log Files

- `app.log` - Main application log with rotating files (10MB max, 5 backups)
- `error/error.log` - Error and critical messages only (5MB max, 3 backups)
- `debug/` - Debug logs and development information
  - `debug.log` - Main debug log (moved from root directory)
  - `test_*.log` - Individual test suite logs
- `access/access.log` - HTTP request access logs (10MB max, 5 backups)

### Log Levels
- **DEBUG**: Detailed information for debugging
- **INFO**: General information about application flow
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for handled exceptions
- **CRITICAL**: Critical errors that may cause application failure

### Log Format
```
%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s
```

### Configuration
Logging is configured in `src/config/logging_config.py` with the following features:
- Rotating file handlers to prevent disk space issues
- Separate handlers for different log levels
- Console output for development
- Structured logging with timestamps and source location

## Usage Examples

### Application Logging
```python
from src.config.logging_config import setup_logging, get_logger

# Setup logging (call once at application start)
setup_logging()

# Get logger for your module
logger = get_logger(__name__)

# Use the logger
logger.info("Application started")
logger.error("Something went wrong")
logger.debug("Debug information")
```

### Test Logging
```python
from pathlib import Path
from src.config.logging_config import setup_test_logging

# Setup test-specific logging
test_log_path = Path("logs/debug/test_my_module.log")
logger = setup_test_logging("test_my_module", test_log_path)

# Use in tests
logger.info("Starting test case")
logger.debug("Test data: %s", test_data)
```

### Usage in Code
```python
from src.config.logging_config import get_logger

logger = get_logger(__name__)
logger.info("This is an info message")
logger.error("This is an error message")
logger.debug("This is a debug message")
```

### Access Logging
For HTTP requests, use the `log_request` function:
```python
from src.config.logging_config import log_request

log_request("GET", "/api/sessions", status_code=200, duration=0.123)
```

### Git Ignore
Log files are automatically ignored by Git (configured in `.gitignore`):
- `logs/*.log`
- `logs/*.txt`
- `logs/*.out`
- `logs/*.err`
- `logs/debug/`
- `logs/error/`
- `logs/access/`

### Monitoring
For production deployments, consider:
- Setting up log rotation monitoring
- Implementing log aggregation (e.g., ELK stack)
- Setting up alerts for error patterns
- Regular cleanup of old log files

## Migration from Old Log Files

If you have existing log files in the project root, use the migration script:

```bash
python scripts/migrate_logs.py
```

This script will:
- Move existing log files to appropriate subdirectories
- Create backups of any conflicting files
- Preserve all log data
- Set up the proper directory structure

### Troubleshooting
1. **No logs appearing**: Check file permissions and disk space
2. **Log files too large**: Adjust rotation settings in `logging_config.py`
3. **Missing debug logs**: Ensure log level is set to DEBUG
4. **Performance issues**: Consider reducing log verbosity in production
5. **Old log files**: Use `scripts/migrate_logs.py` to move legacy log files to the new structure