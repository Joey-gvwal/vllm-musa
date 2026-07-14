# SPDX-License-Identifier: Apache-2.0
"""Cross-repository contract checks for torchada's in-place source porter."""

import sys
from pathlib import Path

import pytest

from build_utils import dependencies

ROOT = Path(__file__).resolve().parent.parent


def test_torchada_floor_is_consistent():
    requirement = dependencies.TORCHADA_REQUIREMENT
    assert requirement == "torchada>=0.1.70"
    assert "dynamic = [\"dependencies\"]" in (ROOT / "pyproject.toml").read_text()
    assert "torchada>=0.1.70" in (
        ROOT / "requirements" / "common.txt"
    ).read_text()
    assert "TORCHADA_REQUIREMENT," in (ROOT / "setup.py").read_text()


def test_musa_image_runtime_dependency_contract():
    private_requirements = (
        ROOT / "requirements" / "musa_private.txt"
    ).read_text().splitlines()
    runtime_requirements = (
        ROOT / "requirements" / "vllm_runtime_transitive.txt"
    ).read_text().splitlines()
    dockerfile = (ROOT / "docker" / "musa.Dockerfile").read_text()

    assert "triton==3.2.0" in private_requirements
    assert "fastapi[standard]" in runtime_requirements
    assert "pycountry" in runtime_requirements
    assert '("triton", "triton", requirement_prefix("triton"))' in dockerfile
    assert '("uvloop", "uvloop", "")' in dockerfile
    assert '("pycountry", "pycountry", "")' in dockerfile


def test_musa_image_stage_and_optional_component_contract():
    dockerfile = (ROOT / "docker" / "musa.Dockerfile").read_text()
    build_script = (ROOT / "docker" / "build_image.sh").read_text()

    stage_markers = (
        "FROM apt_base AS devel",
        "FROM devel AS vllm_musa_deps",
        "FROM vllm_musa_deps AS vllm_musa_installed",
        "FROM vllm_musa_installed AS vllm_rs_build",
        "FROM vllm_musa_installed AS mooncake",
        "FROM mooncake AS final",
    )
    stage_positions = [dockerfile.index(marker) for marker in stage_markers]
    assert stage_positions == sorted(stage_positions)
    assert "FROM apt_base AS runtime" not in dockerfile

    base_stage = dockerfile.split("FROM base AS apt_base", 1)[0]
    for name in (
        "MUSA_HOME",
        "MTGPU_TARGET",
        "TORCH_MUSA_ARCH_LIST",
        "MATE_MUSA_ARCH_LIST",
    ):
        assert name not in base_stage

    deps_stage = dockerfile.split("FROM devel AS vllm_musa_deps", 1)[1].split(
        "FROM vllm_musa_deps AS vllm_musa_installed", 1
    )[0]
    assert "MTGPU_TARGET=mp_31" in deps_stage
    assert "TORCH_MUSA_ARCH_LIST=31" in deps_stage
    assert "MATE_MUSA_ARCH_LIST=3.1" in deps_stage

    mooncake_stage = dockerfile.split(
        "FROM vllm_musa_installed AS mooncake", 1
    )[1].split("FROM mooncake AS final", 1)[0]
    assert "MTHREADS_VISIBLE_DEVICES" not in mooncake_stage
    assert "build.sh --use-mcc" not in mooncake_stage
    assert "cmake .. -DUSE_MUSA=ON -DUSE_ETCD=OFF" in mooncake_stage
    assert "OUTPUT_DIR=dist ./scripts/build_wheel.sh" not in mooncake_stage
    assert "-DSTORE_USE_ETCD=ON" not in mooncake_stage

    assert "ARG BUILD_VLLM_RS=1" in dockerfile
    assert "/tmp/vllm-rs-artifacts/build-mode" in dockerfile
    assert 'BUILD_VLLM_RS="${BUILD_VLLM_RS:-1}"' in build_script
    assert '--build-arg BUILD_VLLM_RS="${BUILD_VLLM_RS}"' in build_script


def test_setup_finds_local_build_helpers_before_importing_them():
    setup = (ROOT / "setup.py").read_text()
    assert setup.index("sys.path.insert(0, str(root))") < setup.index(
        "from build_utils.dependencies import"
    )


def test_archive_vllm_install_uses_upstream_version_override():
    setup = (ROOT / "setup.py").read_text()
    assert 'env.setdefault("VLLM_VERSION_OVERRIDE", "0.24.0")' in setup
    assert "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM" not in setup


def test_stale_torchada_is_upgraded_before_import(monkeypatch):
    observed = []
    versions = iter(["0.1.69", "0.1.70"])

    monkeypatch.setattr(
        dependencies, "_installed_version", lambda _name: next(versions)
    )
    monkeypatch.setattr(
        dependencies,
        "_install",
        lambda requirement: observed.append(("install", requirement)),
    )
    monkeypatch.setattr(
        dependencies.importlib,
        "import_module",
        lambda name: observed.append(("import", name)) or object(),
    )
    monkeypatch.delitem(sys.modules, "torchada", raising=False)

    dependencies.ensure_torchada_installed()

    assert observed == [
        ("install", "torchada>=0.1.70"),
        ("import", "torchada"),
    ]


def test_loaded_stale_torchada_fails_before_mixing_versions(monkeypatch):
    monkeypatch.setattr(dependencies, "_installed_version", lambda _name: "0.1.69")
    monkeypatch.setitem(sys.modules, "torchada", object())

    with pytest.raises(RuntimeError, match="restart the build process"):
        dependencies.ensure_torchada_installed()


def test_no_legacy_mirror_contract_remains():
    legacy_tokens = (
        "csrc_musa",
        "libtorch_stable_musa",
        "attention_musa",
        "quantization_musa",
        "per-file _musa",
    )
    paths = [ROOT / ".gitignore", ROOT / "third_party" / "PINS"]
    paths.extend(sorted((ROOT / "vllm_musa" / "patches" / "series").glob("*.patch")))

    offenders = []
    for path in paths:
        text = path.read_text(errors="replace")
        for token in legacy_tokens:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}: {token}")
    assert not offenders, offenders


def test_native_sampler_includes_flashinfer_header_by_real_name():
    sampler = (ROOT / "csrc" / "musa" / "sampler.mu").read_text()
    assert "#include <flashinfer/sampling.cuh>" in sampler
    assert "#include <flashinfer/sampling.muh>" not in sampler
