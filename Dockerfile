FROM python:3.11-slim AS dependencies


WORKDIR /app
RUN pip install poetry

RUN --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=poetry.lock,target=poetry.lock \
    --mount=type=cache,target=/tmp/cache \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/cache \
    poetry install --no-root --no-ansi

FROM python:3.11-slim
WORKDIR /app

COPY --from=dependencies /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src:/opt/venv/lib/python3.11/site-packages:$PYTHONPATH"

COPY ./src ./src
ARG RELEASE
RUN echo $RELEASE > release

COPY src/ src/

CMD ["python", "src/main.py"]