FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./
COPY src/ src/
RUN uv sync --frozen --no-dev


COPY api/ api/
COPY app/ app/

RUN mkdir -p data models
RUN uv run csurv generate --db data/demo.db
RUN uv run csurv train --db data/demo.db --out models/

EXPOSE 8000 8501

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]