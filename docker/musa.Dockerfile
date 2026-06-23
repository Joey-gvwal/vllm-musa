# vllm-musa image for Ubuntu 22.04 and the MUSA apt stack. Release flow only --
# do not add validation-only wheel/tar download-and-extract logic here.

ARG BASE_IMAGE=ubuntu:22.04
ARG PYTHON_VERSION=3.10

FROM ${BASE_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION

# torch_musa uses "31"; MATE 0.2.3 parses dotted arch tokens such as "3.1".
# MTHREADS_VISIBLE_DEVICES is set only in the final stage: under a MUSA default
# runtime it bind-mounts host driver libs into every build step, breaking apt
# ("Invalid cross-device link").
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_CACHE_DIR=/root/.cache/pip \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MUSA_HOME=/usr/local/musa \
    MTGPU_TARGET=mp_31 \
    TORCH_MUSA_ARCH_LIST=31 \
    MATE_MUSA_ARCH_LIST=3.1
ENV PATH=/usr/local/mtshmem/bin:${MUSA_HOME}/bin:${MUSA_HOME}/mudnn/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/mtshmem/lib:${MUSA_HOME}/lib:${MUSA_HOME}/mudnn/lib:/usr/local/lib

FROM base AS apt_base

ARG PYTHON_VERSION

RUN sed -i 's@http://archive.ubuntu.com/ubuntu/@http://mirrors.aliyun.com/ubuntu/@g' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        g++-12 \
        gcc-12 \
        git \
        git-lfs \
        gnupg \
        infiniband-diags \
        libcurl4-openssl-dev \
        libdrm2 \
        libibverbs-dev \
        libmkl-core \
        libmkl-gnu-thread \
        libmkl-intel-lp64 \
        libnuma1 \
        libomp-dev \
        libopenblas-base \
        libopenmpi-dev \
        librdmacm-dev \
        libssl-dev \
        libstdc++-12-dev \
        libtool \
        libyaml-dev \
        lsb-release \
        lsof \
        make \
        ninja-build \
        numactl \
        openmpi-bin \
        openssh-client \
        patchelf \
        pkg-config \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        python-is-python3 \
        rdma-core \
        unzip \
        xz-utils \
        zip && \
    python -m pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# The torch 2.9.x MUSA wheel links MKL with .so.2 sonames. Ubuntu 22.04 apt
# ships the same logical MKL components without that suffix.
RUN ln -sf /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so \
        /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so.2 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so \
        /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so.2 && \
    ln -sf /usr/lib/x86_64-linux-gnu/libmkl_core.so \
        /usr/lib/x86_64-linux-gnu/libmkl_core.so.2 && \
    ldconfig

FROM apt_base AS runtime

ARG MUSA_APT_SOURCE=https://dl.mthreads.com/repo/repository/ubuntu2204/
ARG INSTALL_MUSA_STACK=auto
ARG MUSA_RUNTIME_VERSION=5.1
ARG MCCL_VERSION=2.3.0
# The muDNN libs (libmudnn3-musa-5*) and mthreads-mtml lack a "-5-1" suffix, so
# pin them by version: muDNN 5.1 is 3.3.0.0 (5.2 would be 3.4.0.0).
ARG MUSA_MUDNN_VERSION=3.3.0.0
ARG MUSA_MTML_VERSION=2.4.1

# Install the MUSA stack from the apt source. MUSA_RUNTIME_VERSION (major.minor,
# e.g. 5.1) is the single version selector and derives the "-5-1" package suffix.
RUN printf 'deb [trusted=true] %s jammy main\n' "${MUSA_APT_SOURCE}" \
        > /etc/apt/sources.list.d/musa.list && \
    if [[ "${INSTALL_MUSA_STACK}" == "0" ]]; then \
        echo "Skipping MUSA apt stack install because INSTALL_MUSA_STACK=0"; \
        exit 0; \
    fi && \
    if [[ "${INSTALL_MUSA_STACK}" == "auto" ]] && command -v mcc >/dev/null 2>&1; then \
        echo "Keeping MUSA stack from BASE_IMAGE"; \
        mcc --version || true; \
        exit 0; \
    fi && \
    apt-get update && \
    if [[ "${MUSA_RUNTIME_VERSION}" =~ ^([0-9]+)\.([0-9]+)(\.|$) ]]; then \
        runtime_major="${BASH_REMATCH[1]}"; \
        runtime_minor="${BASH_REMATCH[2]}"; \
        runtime_suffix="${runtime_major}-${runtime_minor}"; \
    else \
        echo "MUSA_RUNTIME_VERSION must start with <major>.<minor>, got ${MUSA_RUNTIME_VERSION}" >&2; \
        exit 1; \
    fi && \
    resolve_apt_package() { \
        local logical="$1"; \
        local versions="$2"; \
        local allow_unversioned="$3"; \
        shift 3; \
        local spec=""; \
        local version pkg found_version; \
        for version in ${versions}; do \
            for pkg in "$@"; do \
                if ! apt-cache show "${pkg}" >/dev/null 2>&1; then \
                    continue; \
                fi; \
                found_version="$(apt-cache madison "${pkg}" | awk -v w="${version}" '$3 ~ "^" w "([-+~:]|$)" && ex=="" {ex=$3} $3 ~ "^" w "[.]" && pf=="" {pf=$3} END {print (ex!="" ? ex : pf)}')"; \
                if [[ -n "${found_version}" ]]; then \
                    spec="${pkg}=${found_version}"; \
                    break 2; \
                fi; \
            done; \
        done; \
        if [[ -z "${spec}" && "${allow_unversioned}" == "1" ]]; then \
            for pkg in "$@"; do \
                if apt-cache show "${pkg}" >/dev/null 2>&1; then \
                    spec="${pkg}"; \
                    break; \
                fi; \
            done; \
        fi; \
        if [[ -z "${spec}" ]]; then \
            echo "No apt package found for ${logical} with versions [${versions}]; checked: $*" >&2; \
            return 1; \
        fi; \
        echo "${spec}"; \
    } && \
    # Pin the whole MUSA stack to the runtime line and install it in ONE apt
    # transaction: the source now mixes 5.1.0/5.2.0 builds, so an unversioned or
    # per-package install can split it across /usr/local/musa-5.1 and -5.2. The
    # "-5-1" suffix pins most packages; the rest are pinned by version. Entry
    # format: "logical|version-prefixes|candidate,packages" (empty prefixes take
    # the name as-is, else matched against apt-cache madison -- an exact version
    # match wins over a longer one, while a line prefix like 5.1 matches 5.1.0).
    # TODO: replace mccl-s5000 with generic mccl once MCCL ships a unified package.
    musa_pkg_defs=( \
        "musa-toolkit||musa-toolkit-${runtime_suffix}" \
        "musa-toolkit-config||musa-toolkit-${runtime_suffix}-config-common" \
        "mtcc||mtcc-${runtime_suffix}" \
        "musa-musart||musa-musart-${runtime_suffix}" \
        "musa-mupti||musa-mupti-${runtime_suffix}" \
        "musa-mualg||musa-mualg-${runtime_suffix}" \
        "musa-muthrust||musa-muthrust-${runtime_suffix}" \
        "musify||musify-${runtime_suffix}" \
        "libmublas||libmublas-${runtime_suffix}" \
        "libmufft||libmufft-${runtime_suffix}" \
        "libmupp||libmupp-${runtime_suffix}" \
        "libmurand||libmurand-${runtime_suffix}" \
        "libmusparse||libmusparse-${runtime_suffix}" \
        "libmusolver||libmusolver-${runtime_suffix}" \
        "libmublaslt||libmublaslt-${runtime_suffix}" \
        "libmthreads-compute|${MUSA_RUNTIME_VERSION}|libmthreads-compute" \
        "libmudnn3|${MUSA_MUDNN_VERSION}|libmudnn3-musa-${runtime_major}" \
        "libmudnn3-dev|${MUSA_MUDNN_VERSION}|libmudnn3-musa-${runtime_major}-dev" \
        "libmthreads-mtml|${MUSA_MTML_VERSION}|libmthreads-mtml" \
        "mccl-s5000|${MCCL_VERSION}|mccl-s5000" \
    ) && \
    musa_specs=() && \
    for musa_def in "${musa_pkg_defs[@]}"; do \
        IFS='|' read -r musa_logical musa_versions musa_pkgs <<< "${musa_def}"; \
        IFS=',' read -r -a musa_pkg_arr <<< "${musa_pkgs}"; \
        if [[ -n "${musa_versions}" ]]; then musa_allow_unv=0; else musa_allow_unv=1; fi; \
        musa_spec="$(resolve_apt_package "${musa_logical}" "${musa_versions}" "${musa_allow_unv}" "${musa_pkg_arr[@]}")" || exit 1; \
        echo "Pinning ${musa_logical}: ${musa_spec}"; \
        musa_specs+=("${musa_spec}"); \
    done && \
    apt-get install -y --allow-downgrades --no-install-recommends "${musa_specs[@]}" && \
    printf '%s\n' \
        "${MUSA_HOME}/lib" \
        "${MUSA_HOME}/mudnn/lib" \
        "/usr/local/mtshmem/lib" \
        "/usr/lib/x86_64-linux-gnu" \
        > /etc/ld.so.conf.d/musa-runtime.conf && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

# Point /usr/local/musa at whichever /usr/local/musa-* dir actually holds the
# runtime library (libmusart.so.5.*) and register the MUSA lib dirs, in case a
# base image aimed the symlink at a toolkit-less dir. Runs before the shims so
# they act on the right lib dir; a no-op on a correctly set-up base.
RUN real_lib="$(ls /usr/local/musa-*/lib/libmusart.so.5.* 2>/dev/null | sort -V | tail -1)"; \
    if [ -n "${real_lib}" ]; then \
        musa_dir="$(cd "$(dirname "${real_lib}")/.." && pwd)"; \
        if [ "$(readlink -f /usr/local/musa 2>/dev/null)" != "${musa_dir}" ]; then \
            ln -sfn "${musa_dir}" /usr/local/musa; \
            echo "musa-path: repointed /usr/local/musa -> ${musa_dir}"; \
        fi; \
    fi; \
    { echo "${MUSA_HOME}/lib"; \
      echo "${MUSA_HOME}/mudnn/lib"; \
      for d in /usr/local/musa-*/lib /usr/local/musa-*/mudnn/lib; do \
          [ -d "$d" ] && echo "$d"; \
      done; \
      echo /usr/local/mtshmem/lib; \
      echo /usr/lib/x86_64-linux-gnu; } \
      > /etc/ld.so.conf.d/musa-runtime.conf; \
    ldconfig

# --- MUSA 5.1 runtime shims for the MUSA 5.2.0 wheel line ---
# The torch/torch_musa wheels target MUSA 5.2.0 but apt ships 5.1; two 5.1 libs
# otherwise break `import torch`:
#   1. libmupti: 5.1 ships libmupti.so.1.2; the wheels link soname libmupti.so.1.
#   2. libmusolver: the 5.1 build leaves LAPACK zgeqr2_ undefined -- add OpenBLAS
#      as a direct NEEDED of libmusolver only (surgical; torch keeps using MKL).
# Guarded, no-op on a matching 5.2 runtime. Drop once MUSA_RUNTIME_VERSION is 5.2.
RUN musa_lib="${MUSA_HOME}/lib"; \
    mupti_target="$(ls "${musa_lib}"/libmupti.so.1.* 2>/dev/null | sort -V | tail -1)"; \
    if [[ -n "${mupti_target}" && ! -e "${musa_lib}/libmupti.so.1" ]]; then \
        ln -sf "$(basename "${mupti_target}")" "${musa_lib}/libmupti.so.1"; \
        echo "musa5.1-shim: libmupti.so.1 -> $(basename "${mupti_target}")"; \
    fi; \
    solver="$(readlink -f "${musa_lib}/libmusolver.so.1" 2>/dev/null)"; \
    if [[ -n "${solver}" && -e "${solver}" ]] \
        && nm -D "${solver}" 2>/dev/null | grep -qE ' U zgeqr2_?$' \
        && ! objdump -p "${solver}" 2>/dev/null | grep -q 'NEEDED.*libopenblas'; then \
        patchelf --add-needed libopenblas.so.0 "${solver}"; \
        echo "musa5.1-shim: libopenblas.so.0 added to $(basename "${solver}")"; \
    fi; \
    ldconfig

FROM runtime AS mooncake

ARG BUILD_MOONCAKE=1
ARG MOONCAKE_REPO=https://github.com/kvcache-ai/Mooncake.git
ARG MOONCAKE_COMMIT=b6a841dc78c707ec655a563453277d969fb8f38d

ENV PATH=/usr/local/go/bin:${PATH}

# Mooncake is a standalone component; do not route it through the vllm-musa pip
# index args.
RUN if [[ "${BUILD_MOONCAKE}" == "1" ]]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            autoconf \
            ethtool \
            ibverbs-utils \
            openssh-server \
            perftest \
            rdmacm-utils \
            wget && \
        rm -rf /var/lib/apt/lists/* && \
        git clone "${MOONCAKE_REPO}" /workspace/Mooncake && \
        cd /workspace/Mooncake && \
        git checkout "${MOONCAKE_COMMIT}" && \
        git submodule update --init --recursive && \
        bash dependencies.sh -y && \
        mkdir -p build && \
        cd build && \
        cmake .. \
            -DBUILD_UNIT_TESTS=OFF \
            -DUSE_HTTP=ON \
            -DUSE_ETCD=ON \
            -DUSE_MUSA=ON \
            -DSTORE_USE_ETCD=ON \
            -DCMAKE_BUILD_TYPE=Release && \
        cmake --build . -j"$(nproc)" && \
        cmake --install . && \
        cd /workspace/Mooncake && \
        mkdir -p build/mooncake-transfer-engine/nvlink-allocator && \
        cd mooncake-transfer-engine/nvlink-allocator && \
        bash build.sh --use-mcc ../../build/mooncake-transfer-engine/nvlink-allocator/ && \
        cd /workspace/Mooncake && \
        OUTPUT_DIR=dist ./scripts/build_wheel.sh && \
        python -m pip install --no-cache-dir dist/*.whl && \
        rm -rf /workspace/Mooncake /root/.cache/pip /tmp/pip-*; \
    elif [[ "${BUILD_MOONCAKE}" == "0" ]]; then \
        echo "Skipping Mooncake build because BUILD_MOONCAKE=0"; \
    else \
        echo "Unsupported BUILD_MOONCAKE=${BUILD_MOONCAKE}" >&2; \
        exit 1; \
    fi

FROM mooncake AS vllm_musa_deps

# vllm-musa Python deps, installed before the source copy so the layers cache.
# Split across two indexes (per the MUSA release_5.2.0 wheel guide):
#   * PYPI_INDEX_URL (public PyPI): build tools, common.txt runtime deps,
#     transformers, and the MUSA wheels' ordinary deps.
#   * MUSA_PIP_INDEX_URL (internal MT index): the MUSA/MT wheels pinned in
#     requirements/musa_private.txt (torch, torch_musa, MATE, flash_attn_3, ...).
# Some MUSA names (torch, mate, apache-tvm-ffi) also exist on public PyPI, so the
# two indexes must never be merged into one resolve or pip may pick the wrong
# (CUDA/CPU) wheel. URLs live only in docker/build_image.sh and come in as build
# args; a bare `docker build` must supply --build-arg PYPI_INDEX_URL and
# MUSA_PIP_INDEX_URL.
ARG PYPI_INDEX_URL
ARG MUSA_PIP_INDEX_URL

COPY requirements/ /workspace/vllm-musa/requirements/
WORKDIR /workspace/vllm-musa

# 1. MUSA/MT wheels from the internal index only, FIRST and --no-deps (their
#    ordinary deps come in steps 2-3). MUSA torch must land before the public
#    phase: torchada/transformers declare an unpinned `torch`, so a public-first
#    resolve would pull public CUDA torch and the multi-GB nvidia-cuda-* stack.
RUN python -m pip install \
        --no-deps \
        --index-url "${MUSA_PIP_INDEX_URL}" \
        -r requirements/musa_private.txt

# 2. Ordinary third-party wheels from public PyPI: build tools, common.txt, and
#    musa.txt's direct pins (transformers). Step 1 already satisfies `torch`.
RUN musa_public_extras="$(grep -vE '^[[:space:]]*(#|-r[[:space:]]|--|$)' requirements/musa.txt || true)" && \
    python -m pip install \
        --index-url "${PYPI_INDEX_URL}" \
        -r requirements/build.txt \
        -r requirements/common.txt \
        ${musa_public_extras}

# 3. Fill in the MUSA wheels' ordinary deps (sympy, networkx, ...) from public
#    PyPI. Step 1 already pinned every MUSA wheel, so none are re-resolved here.
RUN python -m pip install \
        --index-url "${MUSA_PIP_INDEX_URL}" \
        --extra-index-url "${PYPI_INDEX_URL}" \
        -r requirements/musa_private.txt

FROM vllm_musa_deps AS final

# setup.py's develop_dynamic_library() runs a nested `pip install` for the
# vendored vLLM, pulling vLLM's runtime deps (huggingface-hub, ...) from pip's
# default index. Route those -- and the numpy re-pin below -- through the same
# public index as the deps stage. PYPI_INDEX_URL comes from docker/build_image.sh.
ARG PYPI_INDEX_URL
ENV PIP_INDEX_URL=${PYPI_INDEX_URL}

# Set device visibility only now (see the base-stage note): build stages ran
# without it to keep apt clean; from here `docker run` and the verify below see
# the GPU.
ENV MTHREADS_VISIBLE_DEVICES=all

COPY . /workspace/vllm-musa
RUN python -m pip install \
        -e . --no-build-isolation -v && \
    python -m pip install numpy==1.26

RUN python - <<'PY'
import importlib
import re
from pathlib import Path
from importlib.metadata import version

def requirement_prefix(dist_name):
    pattern = re.compile(rf"^{re.escape(dist_name)}==(.+)$")
    for line in Path("requirements/musa_private.txt").read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1).split("*", 1)[0]
    raise RuntimeError(f"missing {dist_name} pin in requirements/musa_private.txt")

expected = (
    ("numpy", "numpy", "1.26."),
    ("torch", "torch", requirement_prefix("torch")),
    ("torch_musa", "torch_musa", requirement_prefix("torch_musa")),
    ("mate", "mate", ""),
    ("flash_attn_3", "flash_attn_3", ""),
    ("flash_mla", "flash_mla", ""),
    ("deep-gemm", "deep_gemm", ""),
    ("tilelang_musa", "tilelang", ""),
    ("apache-tvm-ffi", "tvm_ffi", ""),
    ("torch_c_dlpack_ext", "torch_c_dlpack_ext", ""),
)

for dist_name, module_name, prefix in expected:
    module = importlib.import_module(module_name)
    installed = version(dist_name)
    if prefix and not installed.startswith(prefix):
        raise RuntimeError(f"{dist_name} expected {prefix}, got {installed}")
    print(f"PASS import {module_name} version={installed}")

for module_name in ("torchada", "vllm", "vllm_musa"):
    module = importlib.import_module(module_name)
    print(f"PASS import {module_name} version={getattr(module, '__version__', 'unknown')}")
PY

RUN rm -rf \
        /root/.cache/pip \
        /root/.cache/vllm-musa \
        /tmp/pip-*

CMD ["/bin/bash"]
