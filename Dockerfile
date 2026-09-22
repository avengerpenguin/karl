# --- Base Stage (Common dependencies) ---
FROM python:3.11-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# --- Dev Stage (Keeps workflow editable) ---
FROM base AS development
# Copy only setup configuration files first to cache dependencies
COPY pyproject.toml setup.py* README.md* ./
RUN pip install --no-cache-dir -e .

# --- Prod Stage (Builds a clean wheel) ---
FROM base AS production
COPY . .
RUN pip install --no-cache-dir .
CMD ["python", "-m", "your_agent_package"]
