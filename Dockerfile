FROM python:3.11.17-slim-bookworm

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter

COPY --from=ghcr.io/astral-sh/uv:0.10.11 /uv /uvx /bin/

ENV PATH="/code/.venv/bin:$PATH" \
    UV_PYTHON="/usr/local/bin/python" \
    UV_PYTHON_DOWNLOADS=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    AWS_LWA_PORT=9696 \
    AWS_LWA_READINESS_CHECK_PATH=/docs

WORKDIR /code

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY model.py predict.py pipeline.bin ./

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
