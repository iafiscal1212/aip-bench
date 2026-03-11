FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install dependencies (core + bench + viz)
RUN poetry install --with bench --with viz --no-root

# Copy project source
COPY . .

# Install the project itself
RUN poetry install --with bench --with viz

# Default: run tests
CMD ["poetry", "run", "pytest", "tests/", "-v", "-m", "not slow"]
