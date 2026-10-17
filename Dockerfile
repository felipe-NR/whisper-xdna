# Whisper encoder on the AMD XDNA1 NPU — reproducible build of the userspace stack.
#
# The image carries everything above the kernel: XRT, MLIR-AIE/IRON, peano and
# this repository. The driver is not in here and cannot be — `amdxdna` is
# in-tree since Linux 6.14 and has to be provided by the host.
#
#   docker build -t npu-whisper .
#   docker run --rm -it --device /dev/accel/accel0 --device /dev/dri \
#              --group-add "$(getent group render | cut -d: -f3)" npu-whisper
#
# Add --build-arg WITH_REFERENCE=1 for the torch/openai-whisper virtualenv used
# to generate the reference tensors, to decode NPU output into text, and to run
# the HTTP service in serve/. It costs about 1.2 GB, which is why it is off by
# default. Add --build-arg WITH_CT2=1 as well for the CTranslate2 decoder
# (NPU_STT_DECODER=ct2, see serve/README.md); compose.ct2.yml builds it so.
#
# Two stages on purpose. Everything is assembled in `build`, which carries a C
# toolchain, git and pip caches, and the final image copies out only the two
# directories that are actually used. Deleting those things in a later layer of
# a single-stage build shrinks nothing: the bytes stay in the layer underneath,
# which is how this image came to be 6.7 GB with 3.6 GB of content.

FROM debian:13 AS build

# The XDNA userspace only exists in trixie-backports. python3-xrt is the
# load-bearing package: without it MLIR-AIE reports "no NPU runtime device is
# available" and the hardware looks absent.
RUN echo "deb http://deb.debian.org/debian trixie-backports main" \
      > /etc/apt/sources.list.d/backports.list \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      python3 python3-pip python3-venv python3-dev \
      build-essential cmake ninja-build \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      -t trixie-backports \
      libxrt-npu2 libxrt-dev libxrt-utils-npu python3-xrt \
 && rm -rf /var/lib/apt/lists/*

# One version for both the checkout and the wheel: setting them separately let
# a --build-arg move one and not the other, which builds a silently mismatched
# image (the compiled aie module comes from the wheel, the designs from git).
ARG MLIR_AIE_VERSION=1.3.4
RUN git clone --depth 1 --branch v${MLIR_AIE_VERSION} \
      https://github.com/Xilinx/mlir-aie.git /opt/mlir-aie

# env_install.sh builds the ironenv virtualenv and pulls llvm-aie (peano).
# It also drags in torch (750 MB) and a doc toolchain that IRON never imports --
# nothing under site-packages/aie references torch -- so they go straight back out.
WORKDIR /opt/mlir-aie
RUN bash -c "source utils/env_install.sh ironenv" \
 && ./ironenv/bin/pip uninstall -y torch triton sympy jedi babel

# The wheel that matches the checkout; env_install leaves a source tree behind
# but the compiled aie python module comes from here.
ARG MLIR_AIE_WHEEL=https://github.com/Xilinx/mlir-aie/releases/download/v${MLIR_AIE_VERSION}/mlir_aie-${MLIR_AIE_VERSION}-cp313-cp313-manylinux_2_35_x86_64.whl
RUN /opt/mlir-aie/ironenv/bin/pip install --no-cache-dir "${MLIR_AIE_WHEEL}"

# Only the matmul design is loaded at run time (rawxrt executes whole_array.py
# out of the checkout); the rest is examples, tests and documentation.
RUN rm -rf /opt/mlir-aie/.git /opt/mlir-aie/test /opt/mlir-aie/mlir_exercises \
           /opt/mlir-aie/programming_guide /opt/mlir-aie/docs \
 && find /opt/mlir-aie/programming_examples -mindepth 1 -maxdepth 1 \
      ! -name basic ! -name utils -exec rm -rf {} + \
 && find /opt/mlir-aie/programming_examples/basic -mindepth 1 -maxdepth 1 \
      ! -name matrix_multiplication -exec rm -rf {} +

# torch and openai-whisper: needed by tools/dump_ref.py, tools/decode_with.py
# and by the serving path in serve/, which runs the decoder in this environment.
# ml_dtypes comes along because rawxrt needs bfloat16 and this venv is where the
# server lives. Off by default: the encoder itself needs none of this.
#
# Installed from requirements/refenv.lock.txt, every package at an exact
# version, with --no-deps: the venv is that list and nothing a resolver picked
# on the day of the build (unpinned, a rebuild drifted from the image the tests
# passed on). triton is not in the list -- a GPU compiler nothing here uses.
COPY requirements/ /tmp/requirements/
ARG WITH_REFERENCE=0
RUN if [ "${WITH_REFERENCE}" = "1" ]; then \
      python3 -m venv /opt/refenv \
   && /opt/refenv/bin/pip install --no-cache-dir --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements/refenv.lock.txt ; \
    else mkdir -p /opt/refenv ; \
    fi

# CTranslate2 for NPU_STT_DECODER=ct2 (serve/ct2_decoder.py), which lives in the
# serving venv, so it needs WITH_REFERENCE=1 too. Versions in
# requirements/refenv-ct2.lock.txt, --no-deps for the same reason.
ARG WITH_CT2=0
RUN if [ "${WITH_CT2}" = "1" ]; then \
      test -x /opt/refenv/bin/pip \
        || { echo "WITH_CT2=1 needs WITH_REFERENCE=1 (refenv)" >&2; exit 1; } \
   && /opt/refenv/bin/pip install --no-cache-dir --no-deps \
        -r /tmp/requirements/refenv-ct2.lock.txt ; \
    fi

# The venv must be exactly the lock files: same packages, same versions,
# nothing extra. A mismatch fails the build here with the diff.
RUN if [ -x /opt/refenv/bin/pip ]; then \
      locks=/tmp/requirements/refenv.lock.txt \
   && if [ "${WITH_CT2}" = "1" ]; then \
        locks="$locks /tmp/requirements/refenv-ct2.lock.txt"; fi \
   && cat $locks | grep -vE '^[[:space:]]*(#|$)' | LC_ALL=C sort -f > /tmp/want \
   && /opt/refenv/bin/pip freeze --all --exclude pip | LC_ALL=C sort -f > /tmp/have \
   && diff -u /tmp/want /tmp/have \
   && echo "refenv matches $locks" ; \
    fi

# --------------------------------------------------------------------------
# The image that ships: XRT, the two virtualenvs and the checkout. No compiler
# toolchain -- kernels are built by peano (clang inside the llvm-aie wheel),
# which came with ironenv.

FROM debian:13-slim

ARG WITH_REFERENCE=0
RUN echo "deb http://deb.debian.org/debian trixie-backports main" \
      > /etc/apt/sources.list.d/backports.list \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates python3 python3-venv \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      -t trixie-backports libxrt-npu2 libxrt-utils-npu python3-xrt \
 && if [ "${WITH_REFERENCE}" = "1" ]; then \
      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ffmpeg ; \
    fi \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/mlir-aie /opt/mlir-aie
# Empty when WITH_REFERENCE=0 -- COPY needs the path to exist either way.
COPY --from=build /opt/refenv /opt/refenv

ENV VIRTUAL_ENV=/opt/mlir-aie/ironenv
ENV PATH=/opt/mlir-aie/ironenv/bin:$PATH
# pyxrt is a distro package and lives outside the virtualenv.
ENV PYTHONPATH=/usr/lib/python3/dist-packages
ENV PEANO_INSTALL_DIR=/opt/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie

# Fail the build here rather than at the first kernel compilation if the
# interpreter version of the base image ever moves off 3.13.
RUN test -d "${PEANO_INSTALL_DIR}" \
 && python3 -c "import aie.iron; import pyxrt; print('iron + pyxrt ok')"

# Same for the ct2 decoder: an import that breaks (numpy/numba, a missing
# converter dependency) fails the build, not the first start of the service.
ARG WITH_CT2=0
RUN if [ "${WITH_CT2}" = "1" ]; then \
      /opt/refenv/bin/python -c "import ctranslate2, whisper, numba, yaml; \
from ctranslate2.converters.converter import Converter; \
print('ctranslate2', ctranslate2.__version__, 'cpu', \
sorted(ctranslate2.get_supported_compute_types('cpu')))" ; \
    fi

# Ties the published package to its repository and licence on registries that
# read OCI labels (GHCR shows the repo, the README and the licence from these).
# revision is the commit the image was built from:
#   docker build --build-arg VCS_REF=$(git rev-parse HEAD) ...
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/drakosha/whisper-xdna" \
      org.opencontainers.image.description="Whisper encoder on the AMD XDNA1 NPU (Ryzen AI, Phoenix) with an HTTP service speaking whisper.cpp's contract" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}"

COPY . /opt/npu-whisper
WORKDIR /opt/npu-whisper

# rawxrt.py resolves its paths relative to the checkout, so it runs here as-is.
RUN if [ -x /opt/refenv/bin/python ]; then \
      ln -sfn /opt/refenv /opt/npu-whisper/refenv ; \
    else rmdir /opt/refenv ; fi

EXPOSE 8090

# Default is a shell: the encoder is a research harness first. `serve/run.sh`
# starts the HTTP service instead — see compose.yml, service `stt`.
CMD ["bash"]
