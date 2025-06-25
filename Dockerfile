# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r aicvgen && useradd -r -g aicvgen aicvgen

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=aicvgen:aicvgen . .

# Create necessary directories for runtime data and grant permissions
# /app/data is for source assets (e.g., prompts)
# /app/instance is for runtime-generated data (logs, sessions, db)
RUN mkdir -p /app/instance/sessions /app/instance/output /app/instance/logs /app/instance/vector_db && \
    chown -R aicvgen:aicvgen /app/instance

# Switch to non-root user
USER aicvgen

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production
# Disable XSRF protection for better compatibility in containerized/proxied environments
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Default command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
