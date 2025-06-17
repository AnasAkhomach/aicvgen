"""Prometheus metrics exporter for the AI CV Generator application.

This module defines and exports metrics for monitoring workflow performance,
LLM usage, and system health.
"""

from prometheus_client import Counter, Histogram, Gauge

# --- Workflow Metrics ---
WORKFLOW_DURATION_SECONDS = Histogram(
    'aicvgen_workflow_duration_seconds',
    'Histogram of CV generation workflow durations.',
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]  # Buckets for different duration ranges
)

WORKFLOW_ERRORS_TOTAL = Counter(
    'aicvgen_workflow_errors_total',
    'Total number of failed CV generation workflows.',
    ['error_type']  # Label to categorize error types
)

WORKFLOW_COMPLETIONS_TOTAL = Counter(
    'aicvgen_workflow_completions_total',
    'Total number of successfully completed CV generation workflows.'
)

# --- LLM Metrics ---
LLM_TOKEN_USAGE_TOTAL = Counter(
    'aicvgen_llm_tokens_total',
    'Total number of LLM tokens used.',
    ['model_name', 'token_type']  # Labels to differentiate by model and token type (input/output)
)

LLM_REQUEST_DURATION_SECONDS = Histogram(
    'aicvgen_llm_request_duration_seconds',
    'Histogram of LLM request durations.',
    ['model_name'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]  # Buckets for LLM response times
)

LLM_REQUESTS_TOTAL = Counter(
    'aicvgen_llm_requests_total',
    'Total number of LLM requests made.',
    ['model_name', 'status']  # Labels for model and request status (success/failure)
)

# --- Agent Metrics ---
AGENT_EXECUTION_DURATION_SECONDS = Histogram(
    'aicvgen_agent_execution_duration_seconds',
    'Histogram of individual agent execution durations.',
    ['agent_name'],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60]  # Buckets for agent execution times
)

AGENT_ERRORS_TOTAL = Counter(
    'aicvgen_agent_errors_total',
    'Total number of agent execution errors.',
    ['agent_name', 'error_type']
)

# --- System Metrics ---
ACTIVE_SESSIONS_GAUGE = Gauge(
    'aicvgen_active_sessions',
    'Number of currently active user sessions.'
)

MEMORY_USAGE_BYTES = Gauge(
    'aicvgen_memory_usage_bytes',
    'Current memory usage in bytes.'
)

# --- Helper Functions ---
def record_workflow_start():
    """Record the start of a workflow execution."""
    # This can be used to track concurrent workflows if needed
    pass

def record_workflow_completion(duration: float):
    """Record successful workflow completion.

    Args:
        duration: Workflow execution duration in seconds
    """
    WORKFLOW_DURATION_SECONDS.observe(duration)
    WORKFLOW_COMPLETIONS_TOTAL.inc()

def record_workflow_error(error_type: str):
    """Record a workflow error.

    Args:
        error_type: Type/category of the error
    """
    WORKFLOW_ERRORS_TOTAL.labels(error_type=error_type).inc()

def record_llm_request(model_name: str, duration: float, input_tokens: int, output_tokens: int, success: bool):
    """Record LLM request metrics.

    Args:
        model_name: Name of the LLM model used
        duration: Request duration in seconds
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        success: Whether the request was successful
    """
    status = 'success' if success else 'failure'

    LLM_REQUEST_DURATION_SECONDS.labels(model_name=model_name).observe(duration)
    LLM_REQUESTS_TOTAL.labels(model_name=model_name, status=status).inc()

    if success:
        LLM_TOKEN_USAGE_TOTAL.labels(model_name=model_name, token_type='input').inc(input_tokens)
        LLM_TOKEN_USAGE_TOTAL.labels(model_name=model_name, token_type='output').inc(output_tokens)

def record_agent_execution(agent_name: str, duration: float, success: bool, error_type: str = None):
    """Record agent execution metrics.

    Args:
        agent_name: Name of the agent
        duration: Execution duration in seconds
        success: Whether the execution was successful
        error_type: Type of error if execution failed
    """
    AGENT_EXECUTION_DURATION_SECONDS.labels(agent_name=agent_name).observe(duration)

    if not success and error_type:
        AGENT_ERRORS_TOTAL.labels(agent_name=agent_name, error_type=error_type).inc()

def update_active_sessions(count: int):
    """Update the active sessions gauge.

    Args:
        count: Current number of active sessions
    """
    ACTIVE_SESSIONS_GAUGE.set(count)

def update_memory_usage(bytes_used: int):
    """Update the memory usage gauge.

    Args:
        bytes_used: Current memory usage in bytes
    """
    MEMORY_USAGE_BYTES.set(bytes_used)
