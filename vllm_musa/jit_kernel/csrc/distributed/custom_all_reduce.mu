#include "../common.h"
#include "../device_utils.h"

#include <musa_runtime.h>
#include <musa_bf16.h>
#include <musa_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

#ifndef SGL_CUSTOM_AR_THREADS
#define SGL_CUSTOM_AR_THREADS 512
#endif

#ifndef SGL_CUSTOM_AR_BLOCKS
#define SGL_CUSTOM_AR_BLOCKS 36
#endif

#ifndef SGL_CUSTOM_AR_VECTOR_LOAD
#define SGL_CUSTOM_AR_VECTOR_LOAD 0
#endif

#ifndef SGL_CUSTOM_AR_ATOMIC_BARRIER
#define SGL_CUSTOM_AR_ATOMIC_BARRIER 1
#endif

#ifndef SGL_CUSTOM_AR_MAX_BLOCKS
#define SGL_CUSTOM_AR_MAX_BLOCKS 120
#endif

constexpr int kMaxBlocks = SGL_CUSTOM_AR_MAX_BLOCKS;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kDefaultThreads = SGL_CUSTOM_AR_THREADS;
constexpr int kDefaultBlockLimit = SGL_CUSTOM_AR_BLOCKS;
constexpr int kMaxRanks = 8;
using FlagType = uint32_t;

struct alignas(128) Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks];
};

struct __align__(16) RankData {
  const void* ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

template <typename T>
__device__ __forceinline__ T downcast_s(float value) {
  return from_float<T>(value);
}

template <>
__device__ __forceinline__ float downcast_s<float>(float value) {
  return value;
}

template <typename T>
__device__ __forceinline__ T& assign_add(T& a, T b) {
  a = downcast_s<T>(to_float(a) + to_float(b));
  return a;
}

template <>
__device__ __forceinline__ float& assign_add<float>(float& a, float b) {
  a += b;
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    assign_add(a.data[i], b.data[i]);
  }
  return a;
}

template <typename T, int N>
__device__ __forceinline__ array_t<float, N> upcast(array_t<T, N> value) {
  if constexpr (std::is_same<T, float>::value) {
    return value;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      out.data[i] = to_float(value.data[i]);
    }
    return out;
  }
}

template <typename O>
__device__ __forceinline__ O downcast(array_t<float, O::size> value) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return value;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; ++i) {
      out.data[i] = downcast_s<typename O::type>(value.data[i]);
    }
    return out;
  }
}

__device__ __forceinline__ void signal_store(FlagType* ptr, FlagType value) {
  volatile_store(static_cast<uint32_t>(value), reinterpret_cast<uint32_t*>(ptr));
}

__device__ __forceinline__ FlagType signal_load(FlagType* ptr) {
  flushInv_byp();
  return static_cast<uint32_t>(volatile_load(reinterpret_cast<uint32_t*>(ptr)));
}

template <int nranks, bool start, bool fence = false>
__device__ __forceinline__ void multi_rank_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  static_assert(!(start && fence));
  if constexpr (!start) {
    __syncthreads_lm();
  }
  if (threadIdx.x < nranks) {
#if SGL_CUSTOM_AR_ATOMIC_BARRIER
    auto flag = atomicAdd(&self_sg->self_counter[blockIdx.x][threadIdx.x], 1) + 1;
    auto* peer = &sg.signals[threadIdx.x]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][threadIdx.x];
    atomicExch(peer, flag);
    while (atomicAdd(local, 0) != flag) {
    }
#else
    auto flag = self_sg->self_counter[blockIdx.x][threadIdx.x] + 1;
    self_sg->self_counter[blockIdx.x][threadIdx.x] = flag;
    auto* peer = &sg.signals[threadIdx.x]->peer_counter[flag & 1][blockIdx.x][rank];
    auto* local = &self_sg->peer_counter[flag & 1][blockIdx.x][threadIdx.x];
    signal_store(peer, flag);
    while (signal_load(local) != flag) {
    }
#endif
  }
  if constexpr (start || fence) {
    __syncthreads_lm();
  }
}

template <typename P, int nranks, typename A>
__device__ __forceinline__ P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < nranks; ++i) {
    packed_assign_add(tmp, upcast(ptrs[i][idx]));
  }
  return downcast<P>(tmp);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_1stage(
    RankData data, RankSignals sg, Signal* self_sg, T* __restrict__ out, int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    reinterpret_cast<P*>(out)[idx] = packed_reduce<P, nranks, A>(reinterpret_cast<const P**>(&data.ptrs[0]), idx);
  }
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename P>
__device__ __forceinline__ P* get_tmp_buf(Signal* signal) {
  return reinterpret_cast<P*>(signal + 1);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_2stage(
    RankData data, RankSignals sg, Signal* self_sg, T* __restrict__ out, int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / nranks;
  int start = rank * part;
  int end = rank == nranks - 1 ? size : start + part;
  int largest_part = part + size % nranks;
  const P* ptrs[nranks];
  P* tmps[nranks];
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    int target = (rank + i) % nranks;
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[target]);
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, nranks, A>(ptrs, idx);
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < nranks; ++i) {
      int gather_rank = (rank + i) % nranks;
      if (gather_rank == nranks - 1 || idx < part) {
        reinterpret_cast<P*>(out)[gather_rank * part + idx] = tmps[i][idx];
      }
    }
  }
}

template <typename T, int nranks, int vlen = 8>
__device__ __forceinline__ void shfl_reduce(float* res) {
  if constexpr (nranks >= 4) {
#pragma unroll
    for (int i = 0; i < vlen; ++i) {
      res[i] += __shfl_xor_sync(0xffffffff, res[i], 16);
    }
  }
#pragma unroll
  for (int i = 0; i < vlen; ++i) {
    res[i] += __shfl_xor_sync(0xffffffff, res[i], 8);
  }
}

template <typename T, int nranks, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) custom_all_reduce_2shot(
    RankData data, RankSignals sg, Signal* self_sg, T* __restrict__ out, int rank, int size) {
  constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
  constexpr int coalesce_num = 8;
  constexpr int coalesce_sft = 3;
  constexpr int group_stride_sft = nranks_sft + coalesce_sft;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
  const int group_id = tid >> group_stride_sft;
  const int coalesce_tid = tid & (coalesce_num - 1);
  const int stride = gridDim.x * blockDim.x;

  typedef int16_t Vec __attribute__((vector_size(16)));
  int idx_base = blockIdx.x * blockDim.x;
  int idx_in_blk = coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  Vec* target_ptr = reinterpret_cast<Vec*>(const_cast<void*>(data.ptrs[target_rank]));
  Vec* buffer_ptr = get_tmp_buf<Vec>(sg.signals[rank]);
  do {
    int idx = idx_in_blk + idx_base;
    float acc[vlen] = {0};
    if (idx < size) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
      Vec raw = target_ptr[idx];
      const T* src = reinterpret_cast<const T*>(&raw);
#else
      const T* src = reinterpret_cast<const T*>(&target_ptr[idx]);
#endif
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] = to_float(src[i]);
      }
    }
    shfl_reduce<T, nranks, vlen>(acc);
    if constexpr (nranks == 8) {
      __shared__ float smem[kMaxThreadsPerBlock << 1];
      if (lane < coalesce_num) {
#pragma unroll
        for (int i = 0; i < vlen; ++i) {
          smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] = acc[i];
        }
      }
      __syncthreads_lm();
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        acc[i] += smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
      }
    }
    if (rank == target_rank && idx < size) {
      Vec res;
#pragma unroll
      for (int i = 0; i < vlen; ++i) {
        reinterpret_cast<T*>(&res)[i] = downcast_s<T>(acc[i]);
      }
      buffer_ptr[idx] = res;
    }
    idx_base += stride;
  } while (idx_base < size);

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);

  buffer_ptr = get_tmp_buf<Vec>(sg.signals[target_rank]);
  idx_in_blk = coalesce_tid + (target_rank << coalesce_sft) + (group_id << group_stride_sft);
  idx_base = blockIdx.x * blockDim.x;
  do {
    int idx = idx_in_blk + idx_base;
    if (idx < size) {
      reinterpret_cast<Vec*>(out)[idx] = buffer_ptr[idx];
    }
    idx_base += stride;
  } while (idx_base < size);
}

template <typename T, int nranks>
void launch_ar(RankData data, RankSignals sg, Signal* self_sg, T* out, int rank, int size, int shot, musaStream_t stream) {
  const int pack = packed_t<T>::P::size;
  TVM_FFI_ICHECK_EQ(size % pack, 0);
  int packed_size = size / pack;
  int blocks = std::min(kDefaultBlockLimit, (packed_size + kDefaultThreads - 1) / kDefaultThreads);
  if (blocks <= 0) {
    return;
  }
  if (shot == 1) {
    cross_device_reduce_1stage<T, nranks><<<blocks, kDefaultThreads, 0, stream>>>(data, sg, self_sg, out, rank, packed_size);
  } else if (shot == 2) {
    if constexpr (std::is_same<T, float>::value) {
      cross_device_reduce_2stage<T, nranks><<<blocks, kDefaultThreads, 0, stream>>>(data, sg, self_sg, out, rank, packed_size);
    } else {
      custom_all_reduce_2shot<T, nranks><<<blocks, kDefaultThreads, 0, stream>>>(data, sg, self_sg, out, rank, packed_size);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "shot must be 1 or 2";
  }
}

template <typename T>
void dispatch_world_size(RankData data, RankSignals sg, Signal* self_sg, T* out, int rank, int world_size, int size, int shot, musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_ar<T, 2>(data, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 4:
      launch_ar<T, 4>(data, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 6:
      launch_ar<T, 6>(data, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 8:
      launch_ar<T, 8>(data, sg, self_sg, out, rank, size, shot, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

}  // namespace

int64_t vllm_musa_custom_ar_meta_size() {
  return static_cast<int64_t>(sizeof(Signal));
}

void vllm_musa_custom_ar_launch_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t shot) {
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(out);
  TVM_FFI_ICHECK_EQ(rank_data.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), out.dtype()));
  TVM_FFI_ICHECK_EQ(tensor_numel(inp), tensor_numel(out));

  RankSignals sg{};
  const auto* ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(ptrs[i]);
  }
  RankData data{};
  const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(out.device());
  const int64_t numel64 = tensor_numel(out);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t nbytes = numel64 * ((static_cast<int64_t>(out.dtype().bits) * out.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);
  const musaError_t copy_err = musaMemcpyAsync(
      reinterpret_cast<void*>(self_buffer_ptr),
      inp.data_ptr(),
      static_cast<size_t>(nbytes),
      musaMemcpyDeviceToDevice,
      stream);
  TVM_FFI_ICHECK_EQ(copy_err, musaSuccess) << "MUSA custom AR copy failed: " << musaGetErrorString(copy_err);
  const int size = static_cast<int>(numel64);

  if (dtype_equal(out.dtype(), dl_float16)) {
    dispatch_world_size(data, sg, self_sg, static_cast<half*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_bfloat16)) {
    dispatch_world_size(data, sg, self_sg, static_cast<__mt_bfloat16*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_float32)) {
    dispatch_world_size(data, sg, self_sg, static_cast<float*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom ar only supports fp16/bf16/fp32";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA custom AR kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_custom_ar_meta_size, vllm_musa_custom_ar_meta_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_custom_ar_launch_unregistered, vllm_musa_custom_ar_launch_unregistered);
