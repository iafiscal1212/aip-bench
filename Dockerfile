FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install package with all optional deps
RUN pip install --no-cache-dir -e ".[all]" && \
    pip install --no-cache-dir pytest

# Default: run tests
CMD ["pytest", "tests/", "-v", "-m", "not slow", "--tb=short"]
