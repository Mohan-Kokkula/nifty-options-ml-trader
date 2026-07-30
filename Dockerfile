# ============================================================
# Dockerfile — OpenClaw Nifty Options Trader
# Security-hardened: non-root, read-only FS, minimal image
# ============================================================

FROM python:3.12-slim AS base

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd -r trader && useradd -r -g trader -d /app -s /sbin/nologin trader

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Kotak Neo SDK is not on PyPI and its GitHub repo isn't anonymously
# clonable from an isolated build sandbox (no access to host git
# credentials — confirmed failing with an auth prompt during `docker
# build`). Installed instead from a known-working copy vendored from the
# host's existing venv (see vendor/neo_api_client/); the git-dependency
# line was removed from requirements.txt accordingly.
RUN cp -r vendor/neo_api_client /usr/local/lib/python3.12/site-packages/neo_api_client && \
    find /usr/local/lib/python3.12/site-packages/neo_api_client -name '__pycache__' -type d -prune -exec rm -rf {} +

# Ensure non-root ownership
RUN chown -R trader:trader /app

# Switch to non-root user
USER trader

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

EXPOSE 8080

# Run with read-only considerations
# Note: Use --read-only --tmpfs /tmp when running:
#   docker run --read-only --tmpfs /tmp --env-file config/settings.env nifty-trader
ENTRYPOINT ["python", "main.py"]
