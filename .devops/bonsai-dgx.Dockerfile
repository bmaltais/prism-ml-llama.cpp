# =============================================================================
# Bonsai 1-bit llama.cpp – DGX Spark / GB10 (Blackwell sm_121)
# Base:  nvcr.io/nvidia/pytorch:25.10-py3  (NGC-optimised for DGX hardware)
# Publicly pullable – no NGC credentials required.
# Multi-stage: heavy PyTorch builder → lean server runtime
# =============================================================================

# Override at build time, e.g.:
#   docker build --build-arg CUDA_DOCKER_ARCH="89;90;121" ...
# When building ON the DGX Spark itself, pass --build-arg GGML_NATIVE=ON to
# enable -march=native for the CPU backend (negligible gain with -ngl 99).
ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.10-py3
# sm_121 = Blackwell (DGX Spark GB10). Separate multiple archs with ';'.
ARG CUDA_DOCKER_ARCH=121
# OFF is correct for CI (build machine ≠ target). ON is safe when building
# directly on DGX Spark.
ARG GGML_NATIVE=OFF

# ── Builder ───────────────────────────────────────────────────────────────────
FROM ${PYTORCH_IMAGE} AS builder

# Re-declare so the ARGs are visible after FROM
ARG CUDA_DOCKER_ARCH
ARG GGML_NATIVE

# The NGC PyTorch image ships cmake, git, gcc/g++ and the full CUDA devel
# toolkit.  We only need to add the headers for LLAMA_CURL and LLAMA_OPENSSL.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl-dev \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the local source tree (this is the Bonsai fork – no git clone needed).
COPY . .

RUN cmake -B build \
    -DLLAMA_OPENSSL=ON \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
    -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j$(nproc)

# Collect all shared libraries required at runtime (ggml backends, etc.)
RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

# ── Runtime ───────────────────────────────────────────────────────────────────
# Re-use the same NGC base so DGX Spark driver compatibility is preserved.
FROM ${PYTORCH_IMAGE} AS runtime

LABEL org.opencontainers.image.source="https://github.com/PrismML-Eng/llama.cpp"
LABEL org.opencontainers.image.description="llama.cpp with 1-bit Bonsai model support (DGX Spark / Blackwell sm_121)"
LABEL org.opencontainers.image.licenses="MIT"

# Runtime deps: libgomp1 (OpenMP), libcurl4 (HF model download at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libcurl4 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ggml backend .so files AND binaries must live in the same directory.
# ggml_backend_load_best() searches: executable's dir → cwd → GGML_BACKEND_DIR.
# /usr/local/lib/ is NOT searched — co-locate everything under /app/.
COPY --from=builder /app/lib/                     /app/

# Compiled server and CLI binaries (same /app/ dir so backends are found)
COPY --from=builder /app/build/bin/llama-server   /app/
COPY --from=builder /app/build/bin/llama-cli      /app/
COPY --from=builder /app/build/bin/llama-quantize /app/

ENV PATH="/app:${PATH}"

# Mount your GGUF model files here at runtime, e.g.:
#   docker run --gpus all -v /path/to/models:/models -p 8080:8080 \
#     <image> --model /models/Bonsai-8B.gguf --ctx-size 65536 -ngl 99
RUN mkdir -p /models
VOLUME ["/models"]

WORKDIR /app
EXPOSE 8080

ENTRYPOINT ["/app/llama-server"]
CMD ["--help"]
