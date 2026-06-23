# Building the vLLM-MUSA image

`docker/build_image.sh` builds the vLLM plugin for Moore Threads (MUSA) GPUs into
a runnable Docker image from `docker/musa.Dockerfile`. It is the supported entry
point — it defines every setting in one place and passes them to the build as
`--build-arg`s, so the Dockerfile itself carries no hardcoded URLs or versions.

The resulting image contains:

- the MUSA runtime SDK (installed from apt),
- the MUSA/MT Python wheels (`torch`, `torch_musa`, `mate`, `flash_attn_3`,
  `flash_mla`, `deep-gemm`, `tilelang_musa`, `apache-tvm-ffi`,
  `torch_c_dlpack_ext`),
- `vllm-musa` and the vendored upstream vLLM, built from source.

## Prerequisites

- **Docker** on the build host.
- **Network access** from the build to:
  - the internal Moore Threads pip index (`MUSA_PIP_INDEX_URL`) — reachable only
    inside the MT network; hosts the MUSA/MT wheels,
  - a public PyPI index/mirror (`PYPI_INDEX_URL`) — ordinary third-party wheels
    and the vendored vLLM's dependencies,
  - the MUSA apt source (`MUSA_APT_SOURCE`) — the runtime SDK,
  - GitHub — the vendored vLLM/flashinfer clones (and Mooncake, if enabled).
- **A MUSA GPU visible to the build** if you want the final-stage import verify to
  pass — see [Building on a MUSA host](#building-on-a-musa-host).

## Quick start

From the repository root:

```bash
bash docker/build_image.sh
```

With the defaults this produces:

```
vllm-musa:ubuntu22.04_py3.10_musa_runtime_5.1_pytorch_2_release_2.9
```

Every setting is an environment variable — override by exporting it or prefixing
the command, e.g.:

```bash
MUSA_RUNTIME_VERSION=5.2 IMAGE_TAG=vllm-musa:dev bash docker/build_image.sh
```

Any extra arguments are forwarded verbatim to `docker build`, so you can also pass
`--build-arg`, `--target`, `--no-cache`, etc.:

```bash
bash docker/build_image.sh --no-cache --build-arg http_proxy=http://proxy:8118
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BASE_IMAGE` | `ubuntu:22.04` | Base image. Point at a local/mirror image if Docker Hub is unreachable, or an `mthreads/musa:*-devel` image to reuse its runtime. |
| `PYTHON_VERSION` | `3.10` | Python version (apt `python3.X`). MUSA 5.2.0 wheels cover 3.10 and 3.12. |
| `MUSA_APT_SOURCE` | `https://dl.mthreads.com/repo/repository/ubuntu2204/` | apt repo for the MUSA runtime SDK. |
| `INSTALL_MUSA_STACK` | `auto` | `auto`: install the MUSA apt stack unless the base already provides `mcc`; `0`: skip (base image supplies the runtime). |
| `MUSA_RUNTIME_VERSION` | `5.1` | MUSA runtime line as `major.minor`; derives apt package names (e.g. `musa-toolkit-5-1`). |
| `MCCL_VERSION` | `2.3.0` | MCCL (collective communication library) version. |
| `PYPI_INDEX_URL` | `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` | Public index for ordinary third-party wheels **and** the vendored vLLM's dependencies. |
| `MUSA_PIP_INDEX_URL` | `https://xxx/simple` | Internal MT index for the MUSA/MT wheels. |
| `BUILD_MOONCAKE` | `0` | `1`: build Mooncake (KV transfer engine) from source; `0`: skip. |
| `MOONCAKE_REPO` / `MOONCAKE_COMMIT` | GitHub / pinned SHA | Mooncake source (only used when `BUILD_MOONCAKE=1`). |
| `IMAGE_REPOSITORY` | `vllm-musa` | Image repository name. |
| `IMAGE_FLAVOR` | `ubuntu22.04_py<py>_musa_runtime_<ver>_pytorch_2_release_<torch>` | Tag flavor; `<torch>` is derived from `requirements/musa_private.txt`. |
| `IMAGE_TAG` | `${IMAGE_REPOSITORY}:${IMAGE_FLAVOR}` | Full image tag. |

## Common scenarios

**Use a specific PyPI mirror** (ordinary wheels + vendored vLLM deps):

```bash
PYPI_INDEX_URL=https://<mirror>/simple bash docker/build_image.sh
```

**Build behind an HTTP proxy** (covers apt, git, and every pip step, including the
nested vLLM install):

```bash
bash docker/build_image.sh \
  --build-arg http_proxy=http://<proxy>:<port> \
  --build-arg https_proxy=http://<proxy>:<port> \
  --build-arg no_proxy=.mthreads.com
```

Keep `no_proxy=.mthreads.com` so the internal MUSA wheel index stays on a direct
connection.

**Docker Hub not reachable** — build from a locally-present base:

```bash
BASE_IMAGE=<local-ubuntu-22.04-image> bash docker/build_image.sh
```

**Use a MUSA 5.2 runtime** (matches the wheel line natively; the 5.1 shims no-op):

```bash
MUSA_RUNTIME_VERSION=5.2 MUSA_APT_SOURCE=<5.2-apt-repo> bash docker/build_image.sh
```

**Include Mooncake:**

```bash
BUILD_MOONCAKE=1 bash docker/build_image.sh
```

**Build only up to the dependency layer** (installs the wheels, skips the vLLM
compile and the import verify — handy for verifying the pip install offline of a
GPU):

```bash
bash docker/build_image.sh --target vllm_musa_deps
```

## Building on a MUSA host

The `final` stage's verify step imports every MUSA package, including `tilelang`
and `flash_mla`, which require `torch.musa.is_available()` to be `True` at import
time. That is only satisfied when the build step can see the GPU — i.e. when the
**MUSA container runtime is the host's default docker runtime**, so `docker build`
`RUN` steps get the device. On a CPU-only builder the build otherwise completes
and then fails at the verify step with
`ImportError: cannot import name 'GPUEvent' from 'tilelang.utils.device'`.

Two build-time details make this work on such a host:

- **Device visibility is set only in the final stage.** `MTHREADS_VISIBLE_DEVICES`
  is deliberately *not* set in the earlier stages: if it were, the MUSA runtime
  would bind-mount host driver libraries into every `apt` step and break package
  installs (`Invalid cross-device link`).
- **MUSA 5.1 shims.** The published wheels are built for MUSA 5.2.0. When building
  against the 5.1 runtime, the runtime stage adds a `libmupti.so.1` soname link
  and a `libmusolver` OpenBLAS dependency so `import torch` works. Both are guarded
  and no-op on a matching 5.2 runtime.

## Verify the built image

```bash
docker run --rm <MUSA GPU flags> \
  vllm-musa:ubuntu22.04_py3.10_musa_runtime_5.1_pytorch_2_release_2.9 \
  python -c "import torch, torch_musa; print('musa available:', torch.musa.is_available())"
```

On a MUSA GPU you should see `musa available: True`.

## How it works (build stages)

`docker/musa.Dockerfile` is multi-stage:

1. **base** — env and library paths.
2. **apt_base** — build toolchain + Python (from apt).
3. **runtime** — the MUSA SDK from apt (`INSTALL_MUSA_STACK`) + the 5.1 shims.
4. **mooncake** — optional Mooncake source build (`BUILD_MOONCAKE`).
5. **vllm_musa_deps** — the Python dependencies, installed in three passes:
   1. MUSA/MT wheels from `MUSA_PIP_INDEX_URL` only (`--no-deps`),
   2. ordinary third-party wheels from `PYPI_INDEX_URL`,
   3. the MUSA wheels' remaining ordinary deps from `PYPI_INDEX_URL`.

   The split keeps names like `torch`/`mate`/`apache-tvm-ffi` resolving from the
   internal index only, so pip never pulls the unrelated public (CUDA) builds.
6. **final** — copies the source, builds `vllm-musa` + the vendored vLLM, re-pins
   `numpy`, and runs the `PASS import ...` verify block.

For the reasoning behind the pip-index split and the runtime shims, see the
comments in `docker/musa.Dockerfile`.
