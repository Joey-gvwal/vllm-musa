# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import torch


def _ensure_numpy_compatible():
    """Ensure numpy<2 (MUSA/torch requirement); the vLLM install can pull numpy>=2."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2", "-q"])


def _ensure_torchada_installed():
    """Ensure torchada is installed (needed for torch.cuda patching)."""
    try:
        import torchada  # noqa: F401
    except ImportError:
        print("Installing torchada...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "torchada", "--upgrade", "-q"]
        )
        import torchada  # noqa: F401


# Run dependency checks at setup start
_ensure_numpy_compatible()
_ensure_torchada_installed()

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

root = Path(__file__).parent.resolve()
sys.path.insert(0, str(root))

from build_utils.ccache import configure_compiler_cache

third_party = Path("third_party")
arch = platform.machine().lower()


def _read_pins():
    """Read third_party/PINS (KEY=VALUE; py3.10 has no tomllib). Shared with
    Makefile.sync so the build and patch-gen pins can't desync."""
    pins = {}
    pins_path = root / "third_party" / "PINS"
    for line in pins_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        pins[key.strip()] = value.split("#", 1)[0].strip()
    return pins


_PINS = _read_pins()

configure_compiler_cache(root)

# Detect editable install (pip install -e .) or develop mode
_is_editable_install = (
    "develop" in sys.argv
    or "editable_wheel" in sys.argv
    or any("--editable" in arg or "-e" in arg for arg in sys.argv)
)

if _is_editable_install:
    Path("vllm").mkdir(exist_ok=True)


def develop_dynamic_library(package_name, source_dir="./", target_override=None):
    """Copy the built vllm.* .so into the editable vLLM clone -- setuptools writes
    them to repo/vllm/ or build/lib*/vllm/, not the clone, so without this
    `import vllm._C` breaks. No-op until the clone + .so exist."""
    try:
        if target_override is not None:
            target_dir = Path(target_override)
        else:
            target_dir = Path(distribution(package_name).locate_file(package_name))
        if not target_dir.is_dir():
            return
        src_dir = Path(source_dir)
        candidates = sorted(
            (
                d
                for d in (src_dir / "vllm", *src_dir.glob("build/lib*/vllm"))
                if d.is_dir() and d.resolve() != target_dir.resolve()
            ),
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        for src in candidates:
            sos = list(src.glob("*.so"))
            if sos:
                for so in sos:
                    shutil.copy2(so, target_dir)
                break
    except PackageNotFoundError:
        print(f"vLLM is not installed '{package_name}'")


class _RepoInfo:
    """Configuration for a third-party git repository."""

    def __init__(self, name, git_repository, git_tag, git_shallow=False):
        self.name = name
        self.git_repository = git_repository
        self.git_tag = git_tag
        self.git_shallow = git_shallow
        self.source_dir = third_party / name


_VLLM_REPO = _RepoInfo(
    name="vllm",
    git_repository="https://github.com/vllm-project/vllm.git",
    git_tag=_PINS["VLLM_TAG"],
    git_shallow=False,
)

_FLASHINFER_REPO = _RepoInfo(
    name="flashinfer",
    git_repository="https://github.com/flashinfer-ai/flashinfer.git",
    git_tag=_PINS["FLASHINFER_COMMIT"],
    git_shallow=False,
)

INCLUDE_DIRS = [
    root / "csrc",
    root / _VLLM_REPO.source_dir / "csrc",
    root / _FLASHINFER_REPO.source_dir / "include",
    root / _FLASHINFER_REPO.source_dir / "csrc",
]

# =============================================================================
# C/C++ Source Files for Extension Modules
# =============================================================================

VLLM_CSRC_SOURCES = [
    str(_VLLM_REPO.source_dir / "csrc/mamba/mamba_ssm/selective_scan_fwd.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cache_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cache_kernels_fused.cu"),
    # paged_attention_v1/v2: CUDA-only, unused on MUSA (mate FlashAttention);
    # skipped + impl-stripped in torch_bindings.cpp (cat-2 patch).
    # str(_VLLM_REPO.source_dir / "csrc/attention/paged_attention_v1.cu"),
    # str(_VLLM_REPO.source_dir / "csrc/attention/paged_attention_v2.cu"),
    str(_VLLM_REPO.source_dir / "csrc/attention/merge_attn_states.cu"),
    str(_VLLM_REPO.source_dir / "csrc/sampler.cu"),
    str(_VLLM_REPO.source_dir / "csrc/topk.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cuda_view.cu"),
    str(
        _VLLM_REPO.source_dir
        / "csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu"
    ),
    str(_VLLM_REPO.source_dir / "csrc/quantization/activation_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/cuda_utils_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/custom_all_reduce.cu"),
    str(_VLLM_REPO.source_dir / "csrc/torch_bindings.cpp"),
    str(_VLLM_REPO.source_dir / "csrc/minimax_reduce_rms_kernel.cu"),
]

VLLM_STABLE_CSRC_SOURCES = [
    # XXX (MUSA): schema-only shim -- the CUDA stable kernels need
    # torch/headeronly/core/Dispatch.h, which is absent from torch_musa.
    str(_VLLM_REPO.source_dir / "csrc/libtorch_stable/torch_bindings.cpp"),
]

VLLM_MUSA_CSRC_SOURCES = [
    "csrc/musa/torch_bindings.cpp",
    "csrc/musa/gemv.mu",
    "csrc/musa/fused_add_rmsnorm.mu",
    "csrc/musa/cache_kernels.mu",
    "csrc/musa/attention/deepseek_v4_cache_store.mu",
    "csrc/musa/attention/deepseek_v4_fused_qkv_rmsnorm.mu",
    "csrc/musa/attention/deepseek_v4_cache_utils.mu",
    "csrc/musa/attention/deepseek_v4_indexer_topk.mu",
    "csrc/musa/attention/deepseek_v4_sparse_flashmla.mu",
    "csrc/musa/attention/deepseek_v4_inv_rope_fp8_quant.mu",
    "csrc/musa/mhc/deepseek_v4_mhc_pre.mu",
    "csrc/musa/moe/deepseek_v4_topk_softplus_sqrt.mu",
    # XXX (MUSA): non-stable -- upstream v0.22 moved this under csrc/libtorch_stable,
    # which needs stable-ABI headers not yet on MUSA.
    "csrc/musa/quantization/per_token_group_quant.cu",
    "csrc/musa/sampler.mu",
    str(_FLASHINFER_REPO.source_dir / "csrc/norm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc/renorm.cu"),
    str(_FLASHINFER_REPO.source_dir / "csrc/sampling.cu"),
]

VLLM_MOE_CSRC_SOURCES = [
    str(_VLLM_REPO.source_dir / "csrc/moe/moe_align_sum_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/moe/topk_softmax_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/moe/topk_softplus_sqrt_kernels.cu"),
    str(_VLLM_REPO.source_dir / "csrc/moe/torch_bindings.cpp"),
]

# =============================================================================
# Compiler and Linker Configuration
# =============================================================================

CXX_FLAGS = ["force_mcc"]
LINK_LIBRARIES = ["c10", "torch", "torch_python", "musart"]
EXTRA_LINK_ARGS = [
    "-Wl,-rpath,$ORIGIN/../../torch/lib",
    f"-L/usr/lib/{arch}-linux-gnu",
    "-lmublasLt",
]

# Detect MTGPU target architecture
DEFAULT_MTGPU_TARGET = "mp_31"
MTGPU_TARGET = os.environ.get("MTGPU_TARGET")

if MTGPU_TARGET:
    print(f"Using MTGPU_TARGET from environment: {MTGPU_TARGET}")
else:
    MTGPU_TARGET = DEFAULT_MTGPU_TARGET

if "MTGPU_TARGET" not in os.environ and torch.musa.is_available():
    try:
        device_props = torch.musa.get_device_properties(0)
        MTGPU_TARGET = f"mp_{device_props.major}{device_props.minor}"
    except Exception as e:
        print(f"Warning: Failed to detect GPU properties: {e}")
elif "MTGPU_TARGET" not in os.environ:
    print(f"Warning: torch.musa not available. Using default target: {MTGPU_TARGET}")

SUPPORTED_MTGPU_TARGETS = ["mp_22", "mp_31"]
if MTGPU_TARGET not in SUPPORTED_MTGPU_TARGETS:
    print(
        f"Warning: Unsupported GPU architecture '{MTGPU_TARGET}'. "
        f"Expected one of: {SUPPORTED_MTGPU_TARGETS}"
    )
    sys.exit(1)

MCC_FLAGS = [
    "-DNDEBUG",
    "-O3",
    "-fPIC",
    "-std=c++17",
    "-x",
    "musa",
    "-mtgpu",
    "-Od3",
    "-ffast-math",
    "-fmusa-flush-denormals-to-zero",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-DUSE_MUSA",
]

if MTGPU_TARGET == "mp_31":
    MCC_FLAGS.append("-DENABLE_FP8")

COMPILE_ARGS = {
    "mcc": MCC_FLAGS,
    "cxx": CXX_FLAGS,
}

# =============================================================================
# Extension Modules
# =============================================================================

EXT_MODULES = [
    CUDAExtension(
        name="vllm._C",
        sources=VLLM_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="vllm._C_stable_libtorch",
        sources=VLLM_STABLE_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="vllm_musa._C",
        sources=VLLM_MUSA_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
    CUDAExtension(
        name="vllm._moe_C",
        sources=VLLM_MOE_CSRC_SOURCES,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=LINK_LIBRARIES,
        extra_link_args=EXTRA_LINK_ARGS,
        py_limited_api=False,
    ),
]


class _CustomBuildExt(BuildExtension):
    """Custom build extension that clones third-party repositories before building."""

    @staticmethod
    def _clone_and_checkout(repo_path, repo_url, git_tag, git_shallow):
        """Clone a git repository and checkout a specific tag/commit."""
        repo_path.parent.mkdir(exist_ok=True)
        if not repo_path.exists():
            clone_cmd = ["git", "clone"]
            if git_shallow:
                clone_cmd += ["--depth", "1"]
            clone_cmd += [repo_url, str(repo_path)]
            subprocess.check_call(clone_cmd)
            subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)
        else:
            subprocess.check_call(["git", "fetch", "--all"], cwd=repo_path)
            subprocess.check_call(["git", "checkout", git_tag], cwd=repo_path)
        subprocess.check_call(["git", "reset", "--hard", git_tag], cwd=repo_path)
        subprocess.check_call(["git", "clean", "-fdx"], cwd=repo_path)

    @staticmethod
    def _install_vllm(repo_path):
        """Install the cloned + patched vLLM (editable) against the existing torch."""
        source_dir = Path(repo_path)

        env = os.environ.copy()
        env["VLLM_TARGET_DEVICE"] = "empty"

        # always editable; compat (path-based .pth) -- the default PEP 660 finder
        # mis-resolves vLLM's submodules and loses to a system vLLM.
        vllm_install_cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(source_dir),
            "--config-settings",
            "editable_mode=compat",
            "--no-build-isolation",
            "-v",
        ]

        steps = [
            {
                "name": "Install vllm use existing torch",
                "cmd": f"cd {source_dir} && python use_existing_torch.py",
                "shell": True,
                "env": None,
            },
            {
                "name": "Install vllm build requirements",
                "cmd": [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(source_dir / "requirements" / "build" / "cuda.txt"),
                ],
                "shell": False,
                "env": None,
            },
            {
                "name": "Install vllm (editable)",
                "cmd": vllm_install_cmd,
                "shell": False,
                "env": env,
            },
        ]

        for step in steps:
            print(f"{step['name']}")

            if step["shell"]:
                print(f"Command: {step['cmd']}")
                subprocess.check_call(step["cmd"], shell=True, env=step["env"])
            else:
                print(f"Command: {' '.join(step['cmd'])}")
                subprocess.check_call(step["cmd"], env=step["env"])

    @staticmethod
    def _apply_musa_patch_series(repo_path):
        """Apply the vLLM-MUSA diff series to the clone at build time (the only
        source-patch mechanism). ``strict`` fails loudly on drift -- regenerate
        via ``make -f Makefile.sync``."""
        if os.environ.get("VLLM_MUSA_NO_BUILD_PATCH", "0") == "1":
            return
        repo = Path(repo_path)
        series = Path(root) / "vllm_musa" / "patches" / "series"
        if not repo.is_dir() or not series.is_dir():
            return
        import importlib.util as _ilu

        ba_path = Path(root) / "vllm_musa" / "patches" / "build_apply.py"
        spec = _ilu.spec_from_file_location("_musa_build_apply", ba_path)
        ba = _ilu.module_from_spec(spec)
        spec.loader.exec_module(ba)
        for name, status in ba.apply_patch_series(repo, series, strict=True):
            print(f"MUSA build patch: {status:16} {name}")

    def run(self):
        if os.environ.get("SKIP_THIRD_PARTY", "0") == "1":
            print("Skipping third-party repositories cloning (SKIP_THIRD_PARTY=1)")
        else:
            print("Cloning third-party repositories...")
            self._clone_and_checkout(
                _VLLM_REPO.source_dir,
                _VLLM_REPO.git_repository,
                _VLLM_REPO.git_tag,
                _VLLM_REPO.git_shallow,
            )
            self._clone_and_checkout(
                _FLASHINFER_REPO.source_dir,
                _FLASHINFER_REPO.git_repository,
                _FLASHINFER_REPO.git_tag,
                _FLASHINFER_REPO.git_shallow,
            )
            print("Third-party repositories ready.")

        # patch the clone BEFORE installing, so the installed vLLM is pre-patched.
        self._apply_musa_patch_series(_VLLM_REPO.source_dir)

        self._install_vllm(_VLLM_REPO.source_dir)

        # Re-ensure numpy<2 after vllm installation (vllm may pull in numpy>=2)
        _ensure_numpy_compatible()

        super().run()


setup(
    ext_modules=EXT_MODULES,
    cmdclass={"build_ext": _CustomBuildExt.with_options(use_ninja=True)},
    include_package_data=False,
    # pinned here because --no-build-isolation skips pyproject.toml deps
    install_requires=[
        "torchada>=0.1.61",
        "mthreads-ml-py>=2.2.11",
        "numpy<2",
        "openai>=2.24.0",
    ],
)

# place the built vllm.* extensions into the editable vLLM clone (see the function).
develop_dynamic_library("vllm", target_override=_VLLM_REPO.source_dir / "vllm")
