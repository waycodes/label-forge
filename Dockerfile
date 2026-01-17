# LabelForge Docker Image
# Based on Ray + vLLM with CUDA support

FROM rayproject/ray:2.53.0-py310-cu121

LABEL maintainer="LabelForge Team"
LABEL description="LabelForge: Deterministic multimodal labeling pipeline"

# Set environment variables for reproducibility
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VLLM_BATCH_INVARIANT=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY pyproject.toml README.md ./

# Install package dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Install vLLM (requires CUDA)
RUN pip install --no-cache-dir vllm>=0.6.5

# Copy source code
COPY labelforge/ ./labelforge/
COPY configs/ ./configs/
COPY prompts/ ./prompts/
COPY docs/ ./docs/
COPY tests/ ./tests/

# Install the package in development mode
RUN pip install --no-cache-dir -e .

# Create directories for runtime
RUN mkdir -p /data /runs /cache

# Set default environment variables
ENV LABELFORGE_CACHE_DIR=/cache \
    LABELFORGE_RUNS_DIR=/runs \
    LABELFORGE_DATA_DIR=/data

# Default command
CMD ["labelforge", "--help"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import labelforge; print(labelforge.__version__)" || exit 1
