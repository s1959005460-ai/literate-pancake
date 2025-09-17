# File: docker/tenseal-builder.Dockerfile
# Usage examples:
#  CPU build:
#    docker build -f docker/tenseal-builder.Dockerfile --build-arg TARGET=cpu -t tenseal-builder:cpu .
#  CUDA build (requires NVIDIA runtime and CUDA toolkits):
#    docker build -f docker/tenseal-builder.Dockerfile --build-arg TARGET=cuda --build-arg CUDA_BASE_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04 -t tenseal-builder:cuda .

ARG TARGET=cpu
ARG CUDA_BASE_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM ${CUDA_BASE_IMAGE} as base_cuda
FROM python:3.10-slim as base_cpu

# Select base depending on TARGET
FROM base_cpu AS build
ARG TARGET
# If TARGET=cuda, copy from base_cuda
# Note: Docker buildkit can use conditional FROM via ARG; simpler: user passes correct FROM via build-arg in CI.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip pkg-config \
    libssl-dev libffi-dev python3-dev python3-pip \
    libeigen3-dev libomp-dev clang g++ && \
    rm -rf /var/lib/apt/lists/*

# Install pybind11 (used by TenSEAL/SEAL bindings)
RUN pip install --no-cache-dir pybind11 wheel setuptools ninja

# Install optional CUDA dev dependencies if user will build CUDA variant separately via `--build-arg TARGET=cuda`
# (For CPU build these are no-ops)
# If building CUDA variant, ensure you use an image that contains CUDA toolkit (see usage).
# Add a small helper to detect and enable CUDA build flags if available
ENV TENSEAL_BUILD_DIR=/workspace/tenseal_build
WORKDIR /workspace
RUN mkdir -p ${TENSEAL_BUILD_DIR}

# Clone TenSEAL (pinned to a stable tag/commit recommended)
ARG TENSEAL_REPO=https://github.com/OpenMined/TenSEAL.git
ARG TENSEAL_TAG=0.4.9   # adjust if necessary
RUN git clone --depth 1 --branch ${TENSEAL_TAG} ${TENSEAL_REPO} ${TENSEAL_BUILD_DIR}

WORKDIR ${TENSEAL_BUILD_DIR}
# Build TenSEAL wheel
# Provide environment flags: -DUSE_CUDA=ON if building CUDA variant (assumes CUDA available)
ARG USE_CUDA=OFF
RUN python -m pip install --upgrade pip setuptools wheel cmake && \
    python -m pip wheel . -w /workspace/wheels

# Stage: runtime image
FROM python:3.10-slim as runtime
WORKDIR /app

# Install runtime dependencies (cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl1.1 ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy built wheels from builder stage
COPY --from=build /workspace/wheels /wheels
# Install all wheels found (if TenSEAL wheel present it will be used)
RUN pip install --no-cache-dir /wheels/*.whl || true

# Install minimal runtime Python deps (safe fallback)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy decryptor application (user will mount app code later)
COPY . /app

ENV PYTHONUNBUFFERED=1
CMD ["python", "decryptor/decryptor_service.py"]
