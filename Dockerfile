# Automated Data Cleaning & Validation System — Production Docker Container
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system utilities if needed (e.g., git, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and raw data files
COPY . /app

# Ensure output directory exists
RUN mkdir -p /app/results /app/output

# Set default entrypoint to pipeline.py CLI
ENTRYPOINT ["python", "pipeline.py"]

# Default CLI arguments: read input dataset listings.csv.gz and write outputs to results/
CMD ["--input", "listings.csv.gz", "--output", "results/"]
