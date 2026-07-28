# TimesFM JAX/PAX side of the equivalence test.
#
# This exists because JAX cannot run the audited checkpoint on the host: jaxlib
# publishes no cp311 Windows wheels and has no CUDA support on Windows. On Linux
# both are available, so the JAX half runs here and hands back a .npz.
#
# The container's job is deliberately small - load a checkpoint, read an array,
# write an array. No benchmark loading, no metrics. Everything else happens on
# the host, so the two runtimes never need to coexist anywhere.
#
#   docker build -t tsfm-timesfm-jax envs/timesfm_jax
#   docker run --rm --gpus all -v "$PWD/artifacts:/artifacts" tsfm-timesfm-jax
#
# Without a working GPU passthrough, drop --gpus all: it falls back to CPU and
# is slow but produces the same numbers, which is all this test needs.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Same timesfm version as the PyTorch side. Comparing one library's two backends
# is a cleaner test than comparing two library versions.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "numpy>=1.26,<2.0" "timesfm[pax]==1.2.9" \
    && pip install --no-cache-dir "jax[cuda12]"

WORKDIR /work
COPY run_equivalence.py /work/run_equivalence.py

# Reads /artifacts/timesfm_equivalence_inputs.npz, writes
# /artifacts/timesfm_forecasts_jax.npz. Mount the host's artifacts directory.
ENTRYPOINT ["python", "/work/run_equivalence.py"]
