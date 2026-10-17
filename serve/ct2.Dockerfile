# The STT image plus CTranslate2, for NPU_STT_DECODER=ct2 (serve/ct2_decoder.py).
#
# An overlay rather than a rebuild: everything below the decoder -- XRT,
# MLIR-AIE/IRON, peano, the refenv with torch and openai-whisper -- is the
# image it extends, which must have been built WITH_REFERENCE=1 (the serving
# path needs refenv). The checkout is copied over the image's own, so serve/
# and src/ are this tree's.
#
#   docker build -f serve/ct2.Dockerfile -t npu-whisper:ct2 .
#   docker build -f serve/ct2.Dockerfile --build-arg BASE=npu-whisper:latest -t npu-whisper:ct2 .

ARG BASE=kostryukov/npu-whisper:1.3.4
FROM ${BASE}

# --no-deps on purpose: ctranslate2 asks for numpy, and letting pip resolve it
# replaces refenv's numpy with one numba (openai-whisper's dependency) refuses
# to import with. pyyaml is the one thing its converter package needs that
# refenv does not already carry.
ARG CT2_VERSION=4.8.2
ARG PYYAML_VERSION=6.0.2
RUN /opt/refenv/bin/pip install --no-cache-dir --no-deps \
      "ctranslate2==${CT2_VERSION}" "pyyaml==${PYYAML_VERSION}" \
 && /opt/refenv/bin/python -c "import ctranslate2, whisper, yaml; \
from ctranslate2.converters.converter import Converter; \
print('ctranslate2', ctranslate2.__version__, 'cpu', \
sorted(ctranslate2.get_supported_compute_types('cpu')))"

COPY . /opt/npu-whisper
# COPY replaces the refenv link the base image made inside the checkout only if
# the tree carries one (.dockerignore keeps refenv/ out); make sure it exists.
RUN ln -sfn /opt/refenv /opt/npu-whisper/refenv

LABEL org.opencontainers.image.description="Whisper encoder on the AMD XDNA1 NPU with a CTranslate2 CPU decoder"
