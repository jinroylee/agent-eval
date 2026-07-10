# agent-eval metric API server.
#
# Default extras keep the image lean (no torch). To bake BERTScore in (multi-GB):
#   docker build --build-arg EXTRAS="server,t2s,bertscore" -t agent-eval-api .
# Closed network: vendor a wheelhouse into the build context (add `COPY wheelhouse /wheels`
# above the pip install line, or use a BuildKit bind mount), then:
#   docker build --build-arg PIP_ARGS="--no-index --find-links=/wheels" -t agent-eval-api .
# Full closed-network recipe: docs/api-server.md.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Everything the wheel build needs (hatchling reads pyproject + README; code lives in src/).
COPY pyproject.toml README.md ./
COPY src ./src

ARG EXTRAS="server,t2s"
ARG PIP_ARGS=""
RUN pip install ${PIP_ARGS} ".[${EXTRAS}]"

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "agent_eval.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
