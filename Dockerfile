FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY backend ./backend
COPY config.toml ./

RUN pip install uv && \
    uv sync --frozen --no-dev

ENV PYTHONPATH=/app/backend/src

CMD ["uv", "run", "python", "backend/watch.py"]
