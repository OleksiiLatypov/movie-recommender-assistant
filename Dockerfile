FROM python:3.12-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency files first for Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies from uv.lock
RUN uv sync --frozen --no-cache --no-install-project

# Copy application source code
COPY . .

# Use project's virtual environment
ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Berlin

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]