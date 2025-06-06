# Logs Directory

This directory contains application logs for the CV AI Generator application.

## Log Structure

### Main Log Files
- `app.log` - Main application log with rotating files (10MB max, 5 backups)
- `error/error.log` - Error and critical messages only (5MB max, 3 backups)
- `debug/debug.log` - Debug level messages (20MB max, 2 backups)
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

### Troubleshooting
1. **No logs appearing**: Check file permissions and disk space
2. **Log files too large**: Adjust rotation settings in `logging_config.py`
3. **Missing debug logs**: Ensure log level is set to DEBUG
4. **Performance issues**: Consider reducing log verbosity in production