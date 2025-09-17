# Dockerfile
# Stage 1: builder for wheels (if TenSEAL needed)
FROM python:3.10-slim AS builder
RUN apt-get update && apt-get install -y build-essential cmake libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY requirements.txt .
RUN pip wheel --no-cache-dir -r requirements.txt -w /workspace/wheels

# Stage 2: runtime
FROM python:3.10-slim
RUN apt-get update && apt-get install -y libssl1.1 ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /workspace/wheels /wheels
RUN pip install --no-index --no-cache-dir /wheels/*.whl || pip install --no-cache-dir -r requirements.txt
COPY . /app
# create user
RUN useradd -m appuser && chown -R appuser /app
USER appuser
ENV PYTHONUNBUFFERED=1
EXPOSE 50051 8080 8000
CMD ["python", "edge/edge_server.py"]
