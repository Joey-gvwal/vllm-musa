from __future__ import annotations

import ctypes
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.utils.torch_utils import direct_register_custom_op

from vllm.logger import init_logger
from vllm_musa.jit_kernel.csrc import allreduce as jit_ar

logger = init_logger(__name__)


cudaError_t = ctypes.c_int


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass(frozen=True)
class _Function:
    name: str
    restype: Any
    argtypes: list[Any]


class _MusaRTLibrary:
    _exported_functions = (
        _Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        _Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        _Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        _Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        _Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        _Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
        _Function("cudaIpcCloseMemHandle", cudaError_t, [ctypes.c_void_p]),
    )
    _library_cache: dict[str, Any] = {}
    _func_cache: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: str = "libmusart.so") -> None:
        if so_file not in self._library_cache:
            self._library_cache[so_file] = ctypes.CDLL(so_file)
        self.lib = self._library_cache[so_file]

        if so_file not in self._func_cache:
            funcs: dict[str, Any] = {}
            for func in self._exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                funcs[func.name] = f
            self._func_cache[so_file] = funcs
        self.funcs = self._func_cache[so_file]

    def check(self, result: cudaError_t) -> None:
        if result != 0:
            error = self.funcs["cudaGetErrorString"](result)
            error_str = error.decode("utf-8") if error else str(result)
            raise RuntimeError(f"MUSART error: {error_str}")

    def malloc(self, size: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.funcs["cudaMalloc"](ctypes.byref(ptr), size))
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        self.check(self.funcs["cudaFree"](ptr))

    def memset(self, ptr: ctypes.c_void_p, value: int, count: int) -> None:
        self.check(self.funcs["cudaMemset"](ptr, value, count))

    def ipc_get_mem_handle(self, ptr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.check(self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), ptr))
        return handle

    def ipc_open_mem_handle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cuda_ipc_mem_lazy_enable_peer_access = 1
        ptr = ctypes.c_void_p()
        self.check(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(ptr), handle, cuda_ipc_mem_lazy_enable_peer_access
            )
        )
        return ptr

    def ipc_close_mem_handle(self, ptr: ctypes.c_void_p) -> None:
        self.check(self.funcs["cudaIpcCloseMemHandle"](ptr))


_COMM_REGISTRY: dict[int, Any] = {}
_NEXT_COMM_ID = 0


def _register_comm(comm: Any) -> int:
    global _NEXT_COMM_ID
    comm_id = _NEXT_COMM_ID
    _NEXT_COMM_ID += 1
    _COMM_REGISTRY[comm_id] = comm
    return comm_id


def _musa_jit_custom_all_reduce(input: torch.Tensor, comm_id: int) -> torch.Tensor:
    comm = _COMM_REGISTRY.get(int(comm_id))
    if comm is None:
        raise RuntimeError(f"MUSA JIT custom all-reduce communicator {comm_id} is gone")
    return comm._custom_all_reduce_impl(input)


def _musa_jit_custom_all_reduce_fake(
    input: torch.Tensor, comm_id: int
) -> torch.Tensor:
    return torch.empty_like(input)


direct_register_custom_op(
    op_name="musa_jit_custom_all_reduce",
    op_func=_musa_jit_custom_all_reduce,
    fake_impl=_musa_jit_custom_all_reduce_fake,
)


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.untyped_storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


@dataclass
class _SharedBuffer:
    pointers: list[int]
    opened_ipc_ptrs: list[int]


def _make_shared_buffer(
    size_in_bytes: int, group: ProcessGroup | None = None
) -> _SharedBuffer:
    lib = _MusaRTLibrary()
    pointer = lib.malloc(size_in_bytes)
    opened_ipc_ptrs: list[int] = []
    try:
        lib.memset(pointer, 0, size_in_bytes)
        handle = lib.ipc_get_mem_handle(pointer)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, handle_i in enumerate(handles):
            if i == rank:
                pointers.append(int(pointer.value))
            else:
                assert isinstance(handle_i, cudaIpcMemHandle_t)
                opened = int(lib.ipc_open_mem_handle(handle_i).value)
                pointers.append(opened)
                opened_ipc_ptrs.append(opened)
        return _SharedBuffer(pointers, opened_ipc_ptrs)
    except Exception:
        for ptr in opened_ipc_ptrs:
            try:
                lib.ipc_close_mem_handle(ctypes.c_void_p(ptr))
            except Exception:
                pass
        try:
            lib.free(pointer)
        except Exception:
            pass
        raise


def _close_ipc_handles(pointers: list[int]) -> None:
    lib = _MusaRTLibrary()
    for ptr in pointers:
        try:
            lib.ipc_close_mem_handle(ctypes.c_void_p(ptr))
        except Exception:
            logger.debug("Failed to close MUSA IPC handle.", exc_info=True)


def _free_own_shared_buffer(
    pointers: list[int], group: ProcessGroup | None = None, rank: int | None = None
) -> None:
    if rank is None:
        rank = dist.get_rank(group=group)
    _MusaRTLibrary().free(ctypes.c_void_p(pointers[rank]))


class _MusaJitCustomAllreduceImpl:
    _SUPPORTED_WORLD_SIZES = (2, 4, 6, 8)
    _MAX_CAR_SIZE = 512 * 1024 * 1024

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = _MAX_CAR_SIZE,
    ) -> None:
        self.disabled = True
        self.group = group
        self.max_size = max_size
        self._IS_CAPTURING = False
        self._comm_id: int | None = None
        self.meta_ptrs: list[int] = []
        self.meta_opened_ipc_ptrs: list[int] = []
        self.buffer_ptrs: list[int] = []
        self.buffer_opened_ipc_ptrs: list[int] = []
        self.buffer_rank_data: torch.Tensor | None = None

        if isinstance(device, int):
            device = torch.device(f"musa:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "MUSA JIT custom allreduce disabled due to unsupported world "
                "size %d. Supported world sizes: %s.",
                self.world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        try:
            meta_bytes = jit_ar.meta_size(self.world_size)
            meta_buffer = _make_shared_buffer(
                meta_bytes + self.max_size, group=group
            )
            self.meta_ptrs = meta_buffer.pointers
            self.meta_opened_ipc_ptrs = meta_buffer.opened_ipc_ptrs
            buffer = _make_shared_buffer(self.max_size, group=group)
            self.buffer_ptrs = buffer.pointers
            self.buffer_opened_ipc_ptrs = buffer.opened_ipc_ptrs
            jit_ar.ensure_compiled(self.world_size)
        except Exception:
            logger.exception("MUSA JIT custom allreduce initialization failed.")
            if self.meta_opened_ipc_ptrs:
                try:
                    _close_ipc_handles(self.meta_opened_ipc_ptrs)
                except Exception:
                    pass
            if self.buffer_opened_ipc_ptrs:
                try:
                    _close_ipc_handles(self.buffer_opened_ipc_ptrs)
                except Exception:
                    pass
            if self.meta_ptrs:
                try:
                    _free_own_shared_buffer(self.meta_ptrs, group=group, rank=self.rank)
                except Exception:
                    pass
            if self.buffer_ptrs:
                try:
                    _free_own_shared_buffer(
                        self.buffer_ptrs, group=group, rank=self.rank
                    )
                except Exception:
                    pass
            self.meta_ptrs = []
            self.meta_opened_ipc_ptrs = []
            self.buffer_ptrs = []
            self.buffer_opened_ipc_ptrs = []
            return

        self.signal_ptrs_cpu = torch.tensor(self.meta_ptrs, dtype=torch.int64)
        buffer_ptrs = self.buffer_ptrs + [0] * (8 - self.world_size)
        self.buffer_rank_data = torch.tensor(buffer_ptrs, dtype=torch.int64)
        self._comm_id = _register_comm(self)
        self.disabled = False
        logger.info_once(
            "Using MUSA JIT custom all-reduce for world_size=%d max_size=%d.",
            self.world_size,
            self.max_size,
        )

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if inp.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0 or inp_size > self.max_size:
            return False
        return _is_weak_contiguous(inp)

    @staticmethod
    def _is_current_stream_capturing() -> bool:
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except Exception:
            return False

    @staticmethod
    def _is_torch_compiling() -> bool:
        try:
            if torch.compiler.is_compiling():
                return True
        except Exception:
            pass
        try:
            return bool(torch._dynamo.is_compiling())
        except Exception:
            return False

    def _custom_all_reduce_custom_op(self, input: torch.Tensor) -> torch.Tensor:
        assert self._comm_id is not None
        return torch.ops.vllm.musa_jit_custom_all_reduce(input, self._comm_id)

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if not self._is_current_stream_capturing():
                # Match vLLM's CustomAllreduce warmup semantics. During graph
                # warmup no communication should run; the real graph capture
                # below records the scratch-buffer copy plus JIT AR kernel.
                return torch.empty_like(input)
            return self._custom_all_reduce_custom_op(input)
        if self._is_torch_compiling():
            # Keep Dynamo/Inductor from tracing into the Python Tile/JIT module
            # loader. The registered op provides the compile-time fake impl and
            # launches the same fixed-staging kernel at runtime.
            return self._custom_all_reduce_custom_op(input)
        # Use fixed staging buffers instead of caching peer IPC pointers for
        # each input. This keeps eager and graph replay on the same
        # pointer-stable kernel.
        return self._custom_all_reduce_impl(input)

    def _custom_all_reduce_impl(self, input: torch.Tensor) -> torch.Tensor:
        assert self.buffer_rank_data is not None
        out = torch.empty_like(input)
        shot = jit_ar.preferred_shot(
            self.world_size, input.numel() * input.element_size()
        )
        jit_ar.launch_unregistered(
            self.buffer_rank_data,
            self.signal_ptrs_cpu,
            input,
            out,
            self.meta_ptrs[self.rank],
            self.buffer_ptrs[self.rank],
            self.max_size,
            self.rank,
            self.world_size,
            shot,
        )
        return out

    def close(self) -> None:
        if not self.disabled and self.meta_opened_ipc_ptrs:
            _close_ipc_handles(self.meta_opened_ipc_ptrs)
        if not self.disabled and self.buffer_opened_ipc_ptrs:
            _close_ipc_handles(self.buffer_opened_ipc_ptrs)
        if not self.disabled and self.meta_ptrs and dist.is_initialized():
            _free_own_shared_buffer(self.meta_ptrs, group=self.group, rank=self.rank)
        if not self.disabled and self.buffer_ptrs and dist.is_initialized():
            _free_own_shared_buffer(self.buffer_ptrs, group=self.group, rank=self.rank)
        if self._comm_id is not None:
            _COMM_REGISTRY.pop(self._comm_id, None)
            self._comm_id = None
        self.meta_ptrs = []
        self.meta_opened_ipc_ptrs = []
        self.buffer_ptrs = []
        self.buffer_opened_ipc_ptrs = []
        self.buffer_rank_data = None
        self.disabled = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MusaJitCustomAllreduce:
    """JIT custom all-reduce for MUSA.

    Keep a single all-reduce implementation across eager and CUDA graph paths.
    Falling back to vLLM's native CustomAllreduce on MUSA can segfault in
    graph/Dynamo startup paths, so unsupported tensors should route to the next
    communicator backend instead of a second custom-allreduce implementation.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size: int = _MusaJitCustomAllreduceImpl._MAX_CAR_SIZE,
        symm_mem_enabled: bool = False,
    ) -> None:
        self.group = group
        self.device = device
        self._jit_comm: _MusaJitCustomAllreduceImpl | None = None
        del symm_mem_enabled

        try:
            self._jit_comm = _MusaJitCustomAllreduceImpl(
                group=group,
                device=device,
                max_size=max_size,
            )
        except Exception:
            logger.exception("Failed to initialize MUSA JIT custom all-reduce.")

        if self._jit_available:
            logger.info_once(
                "Using MUSA JIT custom all-reduce for eager and CUDA graph paths."
            )

    @property
    def _jit_available(self) -> bool:
        return self._jit_comm is not None and not self._jit_comm.disabled

    @property
    def disabled(self) -> bool:
        return not self._jit_available

    @contextmanager
    def capture(self):
        if self._jit_available:
            with self._jit_comm.capture():
                yield
        else:
            yield

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return self._jit_available and self._jit_comm.should_custom_ar(inp)

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        if self._jit_available and self._jit_comm.should_custom_ar(input):
            return self._jit_comm.custom_all_reduce(input)
        return None

    def close(self) -> None:
        if self._jit_comm is not None:
            self._jit_comm.close()
            self._jit_comm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
