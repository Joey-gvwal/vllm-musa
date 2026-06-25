# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source checks for the MUSA QuantFP8 OOT registration."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace


def _stub_module(monkeypatch, name: str) -> ModuleType:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_fp8_with_stubs(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    source_path = (
        repo_root
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "quantization"
        / "fp8.py"
    )

    torch_mod = _stub_module(monkeypatch, "torch")
    torch_mod.Tensor = object
    torch_mod.dtype = object
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.int64 = "int64"
    torch_mod.nn = SimpleNamespace(Module=object)
    torch_mod.cuda = SimpleNamespace(is_available=lambda: False, synchronize=lambda: None)
    torch_mod.inference_mode = nullcontext

    _stub_module(monkeypatch, "vllm")
    _stub_module(monkeypatch, "vllm.model_executor")
    _stub_module(monkeypatch, "vllm.model_executor.layers")
    _stub_module(monkeypatch, "vllm.model_executor.layers.quantization")
    vllm_fp8 = _stub_module(
        monkeypatch, "vllm.model_executor.layers.quantization.fp8"
    )

    class Fp8MoEMethod:
        pass

    Fp8MoEMethod.maybe_roundup_sizes = lambda *args, **kwargs: args[1:3]
    Fp8MoEMethod.create_weights = lambda *args, **kwargs: None
    Fp8MoEMethod.apply = lambda *args, **kwargs: None
    vllm_fp8.Fp8MoEMethod = Fp8MoEMethod

    logger_mod = _stub_module(monkeypatch, "vllm.logger")

    class Logger:
        def info_once(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logger_mod.init_logger = lambda name: Logger()

    fused_moe_mod = _stub_module(monkeypatch, "vllm.model_executor.layers.fused_moe")

    class FusedMoE:
        pass

    fused_moe_mod.FusedMoE = FusedMoE
    fused_moe_mod.fused_experts = lambda **kwargs: "legacy-result"

    platforms_mod = _stub_module(monkeypatch, "vllm.platforms")
    platforms_mod.current_platform = SimpleNamespace(is_musa=lambda: False)

    torch_utils_mod = _stub_module(monkeypatch, "vllm.utils.torch_utils")
    torch_utils_mod.is_torch_equal_or_newer = lambda version: True

    module_name = "musa_fp8_under_test"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quant_fp8_oot_group_quant_uses_musa_helper():
    repo_root = Path(__file__).resolve().parents[1]
    model_executor_init = repo_root / "vllm_musa" / "model_executor" / "__init__.py"
    quant_fp8_oot = (
        repo_root
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "quantization"
        / "input_quant_fp8.py"
    )

    init_source = model_executor_init.read_text()
    source = quant_fp8_oot.read_text()

    assert (
        "import vllm_musa.model_executor.layers.quantization.input_quant_fp8"
        in init_source
    )
    assert "@QuantFP8.register_oot" in source
    assert "class MusaQuantFP8(QuantFP8)" in source
    assert (
        "self.is_group_quant and not self.static and current_platform.is_musa()"
        in source
    )
    assert "if x.dim() != 2:" in source
    assert "fp8_utils.per_token_group_quant_fp8" in source
    assert "return self.forward_native(x, scale, scale_ub, use_triton)" in source


def test_deepseek_v4_defaults_do_not_enable_group_quant_fallback():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "vllm_musa" / "deepseek_v4_fallbacks.py").read_text()

    assert "VLLM_MUSA_ENABLE_TORCH_FP8_GROUP_QUANT_FALLBACK" not in source


def test_musa_fp8_moe_mixed_backend_is_env_gated():
    repo_root = Path(__file__).resolve().parents[1]
    model_executor_init = repo_root / "vllm_musa" / "model_executor" / "__init__.py"
    source = (
        repo_root
        / "vllm_musa"
        / "model_executor"
        / "layers"
        / "quantization"
        / "fp8.py"
    ).read_text()
    warmup_source = (
        repo_root
        / "vllm_musa"
        / "model_executor"
        / "warmup"
        / "moe_autotune.py"
    ).read_text()

    assert 'VLLM_MUSA_FP8_MOE_MIXED_BACKEND' in source
    assert 'VLLM_MUSA_FP8_MOE_AUTOTUNE' in source
    assert 'VLLM_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS' in source
    assert "MusaFp8MoeBucket" in source
    assert "maybe_autotune_musa_fp8_moe_policy" in source
    assert "broadcast_object(policy_data, src=0)" in source
    assert 'def _should_use_musa_mixed_deepgemm(' in source
    assert 'current_platform.is_musa()' in source
    assert '_fp8_backend_name(method) != "DEEPGEMM"' in source
    assert 'return ep_size is not None and ep_size <= 1' in source
    assert 'if _should_use_musa_mixed_deepgemm(self, layer, x):' in source
    assert (
        "import vllm_musa.model_executor.warmup.moe_autotune"
        in model_executor_init.read_text()
    )
    assert "kernel_warmup_module.kernel_warmup = kernel_warmup" in warmup_source
    assert "maybe_autotune_musa_fp8_moe_policy(worker)" in warmup_source


def test_musa_fp8_moe_mixed_backend_gate(monkeypatch):
    fp8 = _load_fp8_with_stubs(monkeypatch)

    class Backend:
        value = "DEEPGEMM"

    method = SimpleNamespace(
        fp8_backend=Backend(),
        is_monolithic=False,
        moe_kernel=object(),
    )
    layer = SimpleNamespace(ep_size=1)
    x = SimpleNamespace(shape=(128, 4096))

    monkeypatch.setattr(fp8, "current_platform", SimpleNamespace(is_musa=lambda: True))
    monkeypatch.delenv("VLLM_MUSA_FP8_MOE_MIXED_BACKEND", raising=False)
    assert not fp8._should_use_musa_mixed_deepgemm(method, layer, x)

    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_MIXED_BACKEND", "1")
    monkeypatch.delenv("VLLM_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS", raising=False)
    assert fp8._should_use_musa_mixed_deepgemm(method, layer, x)

    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS", "129")
    assert not fp8._should_use_musa_mixed_deepgemm(method, layer, x)

    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS", "128")
    method.fp8_backend.value = "TRITON"
    assert not fp8._should_use_musa_mixed_deepgemm(method, layer, x)

    method.fp8_backend.value = "DEEPGEMM"
    layer.ep_size = 4
    assert not fp8._should_use_musa_mixed_deepgemm(method, layer, x)

    layer.ep_size = 1
    monkeypatch.setattr(fp8, "current_platform", SimpleNamespace(is_musa=lambda: False))
    assert not fp8._should_use_musa_mixed_deepgemm(method, layer, x)


def test_musa_fp8_moe_autotune_policy_overrides_fixed_threshold(monkeypatch):
    fp8 = _load_fp8_with_stubs(monkeypatch)

    class Backend:
        value = "DEEPGEMM"

    method = SimpleNamespace(
        fp8_backend=Backend(),
        is_monolithic=False,
        moe_kernel=object(),
    )
    layer = SimpleNamespace(ep_size=1)

    monkeypatch.setattr(fp8, "current_platform", SimpleNamespace(is_musa=lambda: True))
    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_MIXED_BACKEND", "1")
    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_AUTOTUNE", "1")
    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS", "1")
    fp8.set_musa_fp8_moe_bucket_policy(None)

    assert not fp8._should_use_musa_mixed_deepgemm(
        method, layer, SimpleNamespace(shape=(1024, 4096))
    )

    fp8.set_musa_fp8_moe_bucket_policy(
        (
            fp8.MusaFp8MoeBucket(max_tokens=64, backend="triton"),
            fp8.MusaFp8MoeBucket(max_tokens=512, backend="deepgemm"),
            fp8.MusaFp8MoeBucket(max_tokens=1024, backend="triton"),
        )
    )
    assert not fp8._should_use_musa_mixed_deepgemm(
        method, layer, SimpleNamespace(shape=(32, 4096))
    )
    assert fp8._should_use_musa_mixed_deepgemm(
        method, layer, SimpleNamespace(shape=(128, 4096))
    )
    assert not fp8._should_use_musa_mixed_deepgemm(
        method, layer, SimpleNamespace(shape=(2048, 4096))
    )

    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_AUTOTUNE", "0")
    assert fp8._should_use_musa_mixed_deepgemm(
        method, layer, SimpleNamespace(shape=(32, 4096))
    )
    fp8.set_musa_fp8_moe_bucket_policy(None)


def test_musa_fp8_moe_autotune_builds_bucket_policy(monkeypatch):
    fp8 = _load_fp8_with_stubs(monkeypatch)

    class Worker:
        scheduler_config = SimpleNamespace(max_num_batched_tokens=4)

        def get_model(self):
            return object()

    target = SimpleNamespace()

    def fake_measure(layer, backend, num_tokens, warmup, iters):
        del layer, warmup, iters
        if backend == "deepgemm" and num_tokens == 4:
            return 10.0
        if backend == "triton" and num_tokens == 4:
            return 20.0
        if backend == "triton":
            return 10.0
        return 100.0

    monkeypatch.setattr(fp8, "current_platform", SimpleNamespace(is_musa=lambda: True))
    monkeypatch.setattr(fp8, "_MUSA_FP8_MOE_AUTOTUNE_DONE", False)
    monkeypatch.setattr(fp8, "_find_musa_fp8_moe_autotune_target", lambda model: target)
    monkeypatch.setattr(fp8, "_measure_musa_fp8_moe_backend_us", fake_measure)
    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_MIXED_BACKEND", "1")
    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_AUTOTUNE", "1")
    fp8.set_musa_fp8_moe_bucket_policy(None)

    fp8.maybe_autotune_musa_fp8_moe_policy(Worker())

    assert fp8.get_musa_fp8_moe_bucket_policy() == (
        fp8.MusaFp8MoeBucket(max_tokens=2, backend="triton"),
        fp8.MusaFp8MoeBucket(max_tokens=4, backend="deepgemm"),
    )
    fp8.set_musa_fp8_moe_bucket_policy(None)


def test_musa_fp8_moe_mixed_backend_apply_uses_modular_kernel(monkeypatch):
    fp8 = _load_fp8_with_stubs(monkeypatch)

    calls = {}

    class Backend:
        value = "DEEPGEMM"

    class Kernel:
        def apply(self, *args, **kwargs):
            calls["args"] = args
            calls["kwargs"] = kwargs
            return "deepgemm-result"

    method = SimpleNamespace(
        fp8_backend=Backend(),
        is_monolithic=False,
        moe_kernel=Kernel(),
    )
    layer = SimpleNamespace(
        ep_size=1,
        w13_weight="w13",
        w2_weight="w2",
        activation="silu",
        global_num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    x = SimpleNamespace(shape=(128, 4096))

    monkeypatch.setattr(fp8, "current_platform", SimpleNamespace(is_musa=lambda: True))
    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_MIXED_BACKEND", "1")
    monkeypatch.delenv("VLLM_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS", raising=False)

    result = fp8.apply(
        method,
        layer,
        x,
        topk_weights="topk_weights",
        topk_ids="topk_ids",
        shared_experts="shared",
        shared_experts_input="shared_input",
    )

    assert result == "deepgemm-result"
    assert calls["args"] == (
        x,
        "w13",
        "w2",
        "topk_weights",
        "topk_ids",
    )
    assert calls["kwargs"] == {
        "activation": "silu",
        "global_num_experts": 8,
        "expert_map": None,
        "apply_router_weight_on_input": False,
        "shared_experts": "shared",
        "shared_experts_input": "shared_input",
    }


def test_musa_fp8_moe_mixed_backend_apply_keeps_small_token_fallback(monkeypatch):
    fp8 = _load_fp8_with_stubs(monkeypatch)

    class Backend:
        value = "DEEPGEMM"

    class Kernel:
        def apply(self, *args, **kwargs):
            raise AssertionError("small-token path should not use DeepGEMM kernel")

    calls = {}

    def fake_fused_experts(**kwargs):
        calls.update(kwargs)
        return "legacy-result"

    method = SimpleNamespace(
        fp8_backend=Backend(),
        is_monolithic=False,
        moe_kernel=Kernel(),
        mk_can_overlap_shared_experts=False,
        moe_quant_config="quant",
    )
    layer = SimpleNamespace(
        ep_size=1,
        w13_weight="w13",
        w2_weight="w2",
        activation="silu",
        global_num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    x = SimpleNamespace(shape=(127, 4096))

    monkeypatch.setattr(fp8, "current_platform", SimpleNamespace(is_musa=lambda: True))
    monkeypatch.setenv("VLLM_MUSA_FP8_MOE_MIXED_BACKEND", "1")
    monkeypatch.delenv("VLLM_MUSA_FP8_MOE_DEEPGEMM_MIN_TOKENS", raising=False)
    monkeypatch.setattr(fp8, "fused_experts", fake_fused_experts)
    monkeypatch.setattr(fp8, "is_torch_equal_or_newer", lambda _: True)

    result = fp8.apply(
        method,
        layer,
        x,
        topk_weights="topk_weights",
        topk_ids="topk_ids",
    )

    assert result == "legacy-result"
    assert calls["hidden_states"] is x
    assert calls["w1"] == "w13"
    assert calls["w2"] == "w2"
    assert calls["topk_weights"] == "topk_weights"
    assert calls["topk_ids"] == "topk_ids"
    assert calls["quant_config"] == "quant"
