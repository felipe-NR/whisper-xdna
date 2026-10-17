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

# Versions in requirements/refenv-ct2.lock.txt, the same file Dockerfile uses
# with WITH_CT2=1. --no-deps on purpose: ctranslate2 asks for numpy, and
# letting pip resolve it replaces refenv's numpy with one numba
# (openai-whisper's dependency) refuses to import.
COPY requirements/ /tmp/requirements/
RUN /opt/refenv/bin/pip install --no-cache-dir --no-deps \
      -r /tmp/requirements/refenv-ct2.lock.txt \
 && /opt/refenv/bin/python -c "import ctranslate2, whisper, numba, yaml; \
from ctranslate2.converters.converter import Converter; \
print('ctranslate2', ctranslate2.__version__, 'cpu', \
sorted(ctranslate2.get_supported_compute_types('cpu')))"

# The base image's refenv plus the ct2 lock must be exactly the two lock files,
# as in Dockerfile. This is what catches a BASE built from something else.
RUN cat /tmp/requirements/refenv.lock.txt /tmp/requirements/refenv-ct2.lock.txt \
      | grep -vE '^[[:space:]]*(#|$)' | LC_ALL=C sort -f > /tmp/want \
 && /opt/refenv/bin/pip freeze --all --exclude pip | LC_ALL=C sort -f > /tmp/have \
 && diff -u /tmp/want /tmp/have \
 && echo "refenv matches requirements/refenv*.lock.txt"

COPY . /opt/npu-whisper
# COPY replaces the refenv link the base image made inside the checkout only if
# the tree carries one (.dockerignore keeps refenv/ out); make sure it exists.
RUN ln -sfn /opt/refenv /opt/npu-whisper/refenv

# The commit this image was built from: --build-arg VCS_REF=$(git rev-parse HEAD)
ARG VCS_REF=unknown
LABEL org.opencontainers.image.description="Whisper encoder on the AMD XDNA1 NPU with a CTranslate2 CPU decoder" \
      org.opencontainers.image.revision="${VCS_REF}"
