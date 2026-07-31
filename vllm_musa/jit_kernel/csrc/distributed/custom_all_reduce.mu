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

// Limit the TP4 rows=64, hidden=2048 fused two-shot grid to 32 blocks. The
// grid-stride loops still cover all rows while halving block-indexed cross-rank
// synchronization groups relative to the generic 64-block launch.
constexpr int kMaxBlocks = SGL_CUSTOM_AR_MAX_BLOCKS;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kDefaultThreads = SGL_CUSTOM_AR_THREADS;
constexpr int kDefaultBlockLimit = SGL_CUSTOM_AR_BLOCKS;
constexpr int kTp4Rows64TwoShotBlockLimit = 32;
constexpr int kMaxRanks = 8;
using FlagType = uint32_t;

template <int nranks>
inline int fused_ar_rmsnorm_2shot_blocks(int rows, int hidden) {
  int blocks = std::min(rows, kMaxBlocks);
  if constexpr (nranks == 4) {
    if (rows == 64 && hidden == 2048) {
      blocks = std::min(blocks, kTp4Rows64TwoShotBlockLimit);
    }
  }
  return blocks;
}

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

template <typename T, int nranks, bool indirect>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_1stage(
    RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg, T* __restrict__ out, int rank, int size) {
  if constexpr (indirect) {
    data = *data_ptr;
  }
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

template <typename T, int nranks, bool indirect>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) cross_device_reduce_2stage(
    RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg, T* __restrict__ out, int rank, int size) {
  if constexpr (indirect) {
    data = *data_ptr;
  }
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

template <typename T, int nranks, bool indirect, int vlen = 8>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) custom_all_reduce_2shot(
    RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg, T* __restrict__ out, int rank, int size) {
  if constexpr (indirect) {
    data = *data_ptr;
  }
  static_assert((nranks & (nranks - 1)) == 0, "custom_all_reduce_2shot requires power-of-two nranks");
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
template <typename InputT, typename OutputT, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
    cross_device_all_gather_last_dim(RankData data, OutputT* __restrict__ out,
                                     int rows,
                                     int shard_packed_size) {
  using InputP = typename packed_t<InputT>::P;
  using OutputP = array_t<OutputT, InputP::size>;
  const int region_count = rows * nranks;
  if (gridDim.x >= region_count) {
    const int region = blockIdx.x % region_count;
    const int block_in_region = blockIdx.x / region_count;
    const int blocks_per_region = gridDim.x / region_count;
    const int row = region / nranks;
    const int source_rank = region - row * nranks;
    const InputP* source = reinterpret_cast<const InputP*>(data.ptrs[source_rank]) +
                           row * shard_packed_size;
    OutputP* destination = reinterpret_cast<OutputP*>(out) +
                           region * shard_packed_size;
    for (int column = block_in_region * blockDim.x + threadIdx.x;
         column < shard_packed_size;
         column += blocks_per_region * blockDim.x) {
      if constexpr (std::is_same<InputT, OutputT>::value) {
        destination[column] = source[column];
      } else {
        destination[column] = upcast(source[column]);
      }
    }
  } else {
    for (int region = blockIdx.x; region < region_count;
         region += gridDim.x) {
      const int row = region / nranks;
      const int source_rank = region - row * nranks;
      const InputP* source = reinterpret_cast<const InputP*>(data.ptrs[source_rank]) +
                             row * shard_packed_size;
      OutputP* destination = reinterpret_cast<OutputP*>(out) +
                             region * shard_packed_size;
      for (int column = threadIdx.x; column < shard_packed_size;
           column += blockDim.x) {
        if constexpr (std::is_same<InputT, OutputT>::value) {
          destination[column] = source[column];
        } else {
          destination[column] = upcast(source[column]);
        }
      }
    }
  }
}

template <int nranks>
__global__ void cross_device_all_gather_start_barrier(RankSignals sg,
                                                       Signal* self_sg,
                                                       int rank) {
  // Each signalling lane owns its system-scope release/acquire pair.
  if (threadIdx.x < nranks) {
    __threadfence_system();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  if (threadIdx.x < nranks) {
    __threadfence_system();
  }
  __syncthreads_lm();
}

template <int nranks>
__global__ void cross_device_all_gather_end_barrier(RankSignals sg,
                                                     Signal* self_sg,
                                                     int rank) {
  // The same stream reaches this kernel only after every local copy block has
  // completed. Wait for peers before allowing the next staging overwrite.
  if (threadIdx.x < nranks) {
    __threadfence_system();
  }
  __syncthreads_lm();
  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
  if (threadIdx.x < nranks) {
    __threadfence_system();
  }
  __syncthreads_lm();
}

template <typename InputT, typename OutputT, int nranks>
void launch_all_gather_last_dim(RankData data, RankSignals sg,
                                Signal* self_sg, OutputT* out, int rank, int rows,
                                int shard_size, musaStream_t stream) {
  const int pack = packed_t<InputT>::P::size;
  TVM_FFI_ICHECK_EQ(shard_size % pack, 0);
  const int shard_packed_size = shard_size / pack;
  const int region_count = rows * nranks;
  const int block_limit = std::min(kDefaultBlockLimit, kMaxBlocks);
  int blocks = block_limit;
  if (blocks >= region_count) {
    blocks = (blocks / region_count) * region_count;
  }
  if (blocks <= 0) {
    return;
  }
  cross_device_all_gather_start_barrier<nranks>
      <<<1, nranks, 0, stream>>>(sg, self_sg, rank);
  cross_device_all_gather_last_dim<InputT, OutputT, nranks>
      <<<blocks, kDefaultThreads, 0, stream>>>(data, out, rows,
                                               shard_packed_size);
  cross_device_all_gather_end_barrier<nranks>
      <<<1, nranks, 0, stream>>>(sg, self_sg, rank);
}

__device__ __forceinline__ float fused_ar_fast_rsqrt(float value) {
#if ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
  const float half_value = 0.5f * value;
  float y = __frsqrt_rn(value);
  y = y * (1.5f - half_value * y * y);
  return y;
#else
  return rsqrtf(value);
#endif
}

__device__ __forceinline__ float fused_ar_block_sum(float value, float* warp_sums) {
  const int tid = static_cast<int>(threadIdx.x);
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (static_cast<int>(blockDim.x) + 31) >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset, 32);
  }
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads_lm();

  value = tid < num_warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset, 32);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads_lm();
  return warp_sums[0];
}

inline int fused_ar_vec8_block_threads(int hidden) {
  const int vec_count = hidden / 8;
  const int rounded = ((vec_count + 31) / 32) * 32;
  return rounded < 1024 ? rounded : 1024;
}

inline int fused_ar_vec8_2shot_block_threads(int hidden) {
  // MP31 schedules four 32-thread shuffle warps as one 128-thread warp squad.
  // Keep the 2-shot block aligned to that squad while retaining 32-lane shuffles.
  const int vec_count = hidden / 8;
  const int rounded = ((vec_count + 127) / 128) * 128;
  return rounded < kMaxThreadsPerBlock ? rounded : kMaxThreadsPerBlock;
}

template <int nranks>
__device__ __forceinline__ int fused_ar_2shot_owner(int vec_idx, int generic_part) {
  if constexpr ((nranks & (nranks - 1)) == 0) {
    return (vec_idx >> 3) & (nranks - 1);
  } else {
    if (generic_part == 0 || vec_idx >= generic_part * nranks) {
      return nranks - 1;
    }
    return vec_idx / generic_part;
  }
}

template <typename WT>
__device__ __forceinline__ float load_weight_scalar(const WT* weight, int idx) {
  return to_float(weight[idx]);
}

template <>
__device__ __forceinline__ float load_weight_scalar<float>(const float* weight, int idx) {
  return weight[idx];
}

// Direct TP2 one-stage kernel. It keeps the reduced or residual value in
// registers across the RMS reduction, avoiding the second global read in the
// generic vLLM kernel. The specialized path uses one vec8 lane per thread for
// both TP2 h2048 vector-rank and non-power-of-two h5120 register-cache cases.
template <
    typename T,
    typename WT,
    bool HasResidual,
    bool WriteReduced>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
fused_ar_rmsnorm_tp2_specialized_kernel(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual,
    const WT* __restrict__ weight,
    T* __restrict__ norm_out,
    T* __restrict__ residual_out,
    T* __restrict__ reduced,
    int rank,
    int rows,
    int hidden,
    float eps) {
  if (device_data != nullptr) {
    data = *device_data;
  }
  using P = typename packed_t<T>::P;
  static_assert(P::size == 8, "specialized TP2 fused path expects 8 values per pack");
  extern __shared__ float warp_sums[];
  const int tid = static_cast<int>(threadIdx.x);
  const int packed_hidden = hidden / P::size;
  const int packed_col = tid;

  multi_rank_barrier<2, true>(sg, self_sg, rank);

  for (int row = static_cast<int>(blockIdx.x);
       row < rows;
       row += static_cast<int>(gridDim.x)) {
    float values[P::size] = {0.0f};
    float square_sum = 0.0f;
    int pack_idx = 0;

    if (row < rows) {
      pack_idx = row * packed_hidden + packed_col;
      const P local_pack =
          reinterpret_cast<const P*>(data.ptrs[rank])[pack_idx];
      const P peer_pack =
          reinterpret_cast<const P*>(data.ptrs[rank ^ 1])[pack_idx];
      P reduced_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        reduced_pack.data[i] = from_float<T>(
            to_float(local_pack.data[i]) + to_float(peer_pack.data[i]));
      }
      if constexpr (WriteReduced) {
        reinterpret_cast<P*>(reduced)[pack_idx] = reduced_pack;
      }

      P norm_input_pack = reduced_pack;
      if constexpr (HasResidual) {
        const P residual_pack =
            reinterpret_cast<const P*>(residual)[pack_idx];
#pragma unroll
        for (int i = 0; i < P::size; ++i) {
          norm_input_pack.data[i] = from_float<T>(
              to_float(reduced_pack.data[i]) +
              to_float(residual_pack.data[i]));
        }
        reinterpret_cast<P*>(residual_out)[pack_idx] = norm_input_pack;
      }

#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        values[i] = to_float(norm_input_pack.data[i]);
        square_sum += values[i] * values[i];
      }
    }

    const float row_square_sum = fused_ar_block_sum(square_sum, warp_sums);
    const float scale = fused_ar_fast_rsqrt(
        row_square_sum / static_cast<float>(hidden) + eps);

    if (row < rows) {
      const int col = packed_col * P::size;
      P out_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        const float gamma = load_weight_scalar<WT>(weight, col + i);
        out_pack.data[i] = from_float<T>(values[i] * scale * gamma);
      }
      reinterpret_cast<P*>(norm_out)[pack_idx] = out_pack;
    }
  }

  multi_rank_barrier<2, false>(sg, self_sg, rank);
}

template <typename T, typename WT, bool HasResidual, bool WriteReduced>
bool launch_fused_ar_rmsnorm_tp2_specialized(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual,
    const WT* weight,
    T* norm_out,
    T* residual_out,
    T* reduced,
    int rank,
    int rows,
    int hidden,
    float eps,
    musaStream_t stream) {
  if (rows <= 0) {
    return false;
  }

  if (hidden == 2048 && rows <= 128) {
    // Use one vec8 lane per thread (2048 / 8 == 256 threads), rather than the
    // generic 512-thread module.
    constexpr int threads = 2048 / 8;
    const int blocks = std::min(kMaxBlocks, rows);
    const size_t smem_bytes =
        static_cast<size_t>((threads + 31) / 32) * sizeof(float);
    fused_ar_rmsnorm_tp2_specialized_kernel<
        T, WT, HasResidual, WriteReduced>
        <<<blocks, threads, smem_bytes, stream>>>(
            data, device_data, sg, self_sg, residual, weight, norm_out,
            residual_out, reduced, rank, rows, hidden, eps);
    return true;
  }

  if (hidden == 5120 && rows <= 128) {
    constexpr int threads = 5120 / 8;
    const int blocks = std::min(kMaxBlocks, rows);
    const size_t smem_bytes =
        static_cast<size_t>((threads + 31) / 32) * sizeof(float);
    fused_ar_rmsnorm_tp2_specialized_kernel<
        T, WT, HasResidual, WriteReduced>
        <<<blocks, threads, smem_bytes, stream>>>(
            data, device_data, sg, self_sg, residual, weight, norm_out,
            residual_out, reduced, rank, rows, hidden, eps);
    return true;
  }

  return false;
}

template <typename T, typename WT, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) fused_ar_rmsnorm_1stage_kernel(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const WT* __restrict__ weight,
    T* __restrict__ norm_out,
    T* __restrict__ reduced,
    int rank,
    int rows,
    int hidden,
    float eps) {
  if (device_data != nullptr) {
    data = *device_data;
  }
  using P = typename packed_t<T>::P;
  static_assert(P::size == 8, "fused AR-RMSNorm currently expects 8 values per 16B pack");
  extern __shared__ float warp_sums[];
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / P::size;

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  for (int row = static_cast<int>(blockIdx.x); row < rows; row += static_cast<int>(gridDim.x)) {
    const int row_pack_base = row * vec_count;
    float square_sum = 0.0f;

    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
      const int pack_idx = row_pack_base + vec_idx;
      float acc[P::size];
      const P first = reinterpret_cast<const P*>(data.ptrs[0])[pack_idx];
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        acc[i] = to_float(first.data[i]);
      }
#pragma unroll
      for (int r = 1; r < nranks; ++r) {
        const P peer = reinterpret_cast<const P*>(data.ptrs[r])[pack_idx];
#pragma unroll
        for (int i = 0; i < P::size; ++i) {
          acc[i] += to_float(peer.data[i]);
        }
      }

      P reduced_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        reduced_pack.data[i] = from_float<T>(acc[i]);
        const float x = to_float(reduced_pack.data[i]);
        square_sum += x * x;
      }
      reinterpret_cast<P*>(reduced)[pack_idx] = reduced_pack;
    }

    square_sum = fused_ar_block_sum(square_sum, warp_sums);
    const float scale = fused_ar_fast_rsqrt(square_sum / static_cast<float>(hidden) + eps);

    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
      const int pack_idx = row_pack_base + vec_idx;
      const int col = vec_idx * P::size;
      const P reduced_pack = reinterpret_cast<const P*>(reduced)[pack_idx];
      P out_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        const float x = to_float(reduced_pack.data[i]);
        const float w = load_weight_scalar<WT>(weight, col + i);
        out_pack.data[i] = from_float<T>(x * scale * w);
      }
      reinterpret_cast<P*>(norm_out)[pack_idx] = out_pack;
    }
  }

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) fused_ar_rmsnorm_reduce_scatter_store_kernel(
    RankData data,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    int rows,
    int hidden) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  static_assert(P::size == 8, "fused AR-RMSNorm currently expects 8 values per 16B pack");
  const int tid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  const int vec_count = hidden / P::size;
  const int packed_size = rows * vec_count;
  const int part = packed_size / nranks;
  const int start = rank * part;
  const int end = rank == nranks - 1 ? packed_size : start + part;
  const P* ptrs[nranks];
  P* tmps[nranks];
#pragma unroll
  for (int i = 0; i < nranks; ++i) {
    ptrs[i] = reinterpret_cast<const P*>(data.ptrs[i]);
    tmps[i] = get_tmp_buf<P>(sg.signals[i]);
  }

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);
  for (int idx = start + tid; idx < end; idx += stride) {
    const P reduced_pack = packed_reduce<P, nranks, A>(ptrs, idx);
#pragma unroll
    for (int dst = 0; dst < nranks; ++dst) {
      tmps[dst][idx] = reduced_pack;
    }
  }
  __musa_barrier_slc();
  __syncthreads_lm();
  if (threadIdx.x == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);
}

template <typename T, typename WT>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) fused_ar_rmsnorm_local_tmp_kernel(
    RankSignals sg,
    const WT* __restrict__ weight,
    T* __restrict__ norm_out,
    T* __restrict__ reduced,
    int rank,
    int rows,
    int hidden,
    float eps) {
  using P = typename packed_t<T>::P;
  static_assert(P::size == 8, "fused AR-RMSNorm currently expects 8 values per 16B pack");
  extern __shared__ float warp_sums[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / P::size;
  const int row_pack_base = row * vec_count;
  const P* tmps = get_tmp_buf<P>(sg.signals[rank]);
  float square_sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
    const int pack_idx = row_pack_base + vec_idx;
    const P reduced_pack = tmps[pack_idx];
    reinterpret_cast<P*>(reduced)[pack_idx] = reduced_pack;
#pragma unroll
    for (int i = 0; i < P::size; ++i) {
      const float x = to_float(reduced_pack.data[i]);
      square_sum += x * x;
    }
  }

  square_sum = fused_ar_block_sum(square_sum, warp_sums);
  const float scale = fused_ar_fast_rsqrt(square_sum / static_cast<float>(hidden) + eps);

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
    const int pack_idx = row_pack_base + vec_idx;
    const int col = vec_idx * P::size;
    const P reduced_pack = reinterpret_cast<const P*>(reduced)[pack_idx];
      P out_pack;
#pragma unroll
    for (int i = 0; i < P::size; ++i) {
      const float x = to_float(reduced_pack.data[i]);
      const float w = load_weight_scalar<WT>(weight, col + i);
      out_pack.data[i] = from_float<T>(x * scale * w);
    }
    reinterpret_cast<P*>(norm_out)[pack_idx] = out_pack;
  }
}

template <typename T, typename WT, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) fused_ar_residual_rmsnorm_1stage_kernel(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual,
    const WT* __restrict__ weight,
    T* __restrict__ norm_out,
    T* __restrict__ residual_out,
    T* __restrict__ reduced,
    int rank,
    int rows,
    int hidden,
    float eps) {
  if (device_data != nullptr) {
    data = *device_data;
  }
  using P = typename packed_t<T>::P;
  static_assert(P::size == 8, "fused AR-residual-RMSNorm currently expects 8 values per 16B pack");
  extern __shared__ float warp_sums[];
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / P::size;

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  for (int row = static_cast<int>(blockIdx.x); row < rows; row += static_cast<int>(gridDim.x)) {
    const int row_pack_base = row * vec_count;
    float square_sum = 0.0f;

    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
      const int pack_idx = row_pack_base + vec_idx;
      float acc[P::size];
      const P first = reinterpret_cast<const P*>(data.ptrs[0])[pack_idx];
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        acc[i] = to_float(first.data[i]);
      }
#pragma unroll
      for (int r = 1; r < nranks; ++r) {
        const P peer = reinterpret_cast<const P*>(data.ptrs[r])[pack_idx];
#pragma unroll
        for (int i = 0; i < P::size; ++i) {
          acc[i] += to_float(peer.data[i]);
        }
      }

      P reduced_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        reduced_pack.data[i] = from_float<T>(acc[i]);
      }
      reinterpret_cast<P*>(reduced)[pack_idx] = reduced_pack;
      const P residual_pack = reinterpret_cast<const P*>(residual)[pack_idx];
      P residual_out_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        const float x = to_float(reduced_pack.data[i]) + to_float(residual_pack.data[i]);
        residual_out_pack.data[i] = from_float<T>(x);
        const float stored = to_float(residual_out_pack.data[i]);
        square_sum += stored * stored;
      }
      reinterpret_cast<P*>(residual_out)[pack_idx] = residual_out_pack;
    }

    square_sum = fused_ar_block_sum(square_sum, warp_sums);
    const float scale = fused_ar_fast_rsqrt(square_sum / static_cast<float>(hidden) + eps);

    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
      const int pack_idx = row_pack_base + vec_idx;
      const int col = vec_idx * P::size;
      const P residual_out_pack = reinterpret_cast<const P*>(residual_out)[pack_idx];
      P out_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        const float x = to_float(residual_out_pack.data[i]);
        const float w = load_weight_scalar<WT>(weight, col + i);
        out_pack.data[i] = from_float<T>(x * scale * w);
      }
      reinterpret_cast<P*>(norm_out)[pack_idx] = out_pack;
    }
  }

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, typename WT>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) fused_ar_residual_rmsnorm_local_tmp_kernel(
    RankSignals sg,
    const T* __restrict__ residual,
    const WT* __restrict__ weight,
    T* __restrict__ norm_out,
    T* __restrict__ residual_out,
    T* __restrict__ reduced,
    int rank,
    int rows,
    int hidden,
    float eps) {
  using P = typename packed_t<T>::P;
  static_assert(P::size == 8, "fused AR-residual-RMSNorm currently expects 8 values per 16B pack");
  extern __shared__ float warp_sums[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / P::size;
  const int row_pack_base = row * vec_count;
  const P* tmps = get_tmp_buf<P>(sg.signals[rank]);
  float square_sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
    const int pack_idx = row_pack_base + vec_idx;
    const P reduced_pack = tmps[pack_idx];
    reinterpret_cast<P*>(reduced)[pack_idx] = reduced_pack;
    const P residual_pack = reinterpret_cast<const P*>(residual)[pack_idx];
    P residual_out_pack;
#pragma unroll
    for (int i = 0; i < P::size; ++i) {
      const float x = to_float(reduced_pack.data[i]) + to_float(residual_pack.data[i]);
      residual_out_pack.data[i] = from_float<T>(x);
      const float stored = to_float(residual_out_pack.data[i]);
      square_sum += stored * stored;
    }
    reinterpret_cast<P*>(residual_out)[pack_idx] = residual_out_pack;
  }

  square_sum = fused_ar_block_sum(square_sum, warp_sums);
  const float scale = fused_ar_fast_rsqrt(square_sum / static_cast<float>(hidden) + eps);

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
    const int pack_idx = row_pack_base + vec_idx;
    const int col = vec_idx * P::size;
    const P residual_out_pack = reinterpret_cast<const P*>(residual_out)[pack_idx];
    P out_pack;
#pragma unroll
    for (int i = 0; i < P::size; ++i) {
      const float x = to_float(residual_out_pack.data[i]);
      const float w = load_weight_scalar<WT>(weight, col + i);
      out_pack.data[i] = from_float<T>(x * scale * w);
    }
    reinterpret_cast<P*>(norm_out)[pack_idx] = out_pack;
  }
}

template <typename T, typename WT, int nranks>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) fused_ar_residual_rmsnorm_no_raw_1stage_kernel(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual,
    const WT* __restrict__ weight,
    T* __restrict__ norm_out,
    T* __restrict__ residual_out,
    int rank,
    int rows,
    int hidden,
    float eps) {
  if (device_data != nullptr) {
    data = *device_data;
  }
  using P = typename packed_t<T>::P;
  static_assert(P::size == 8, "fused AR-residual-RMSNorm no-raw currently expects 8 values per 16B pack");
  extern __shared__ float warp_sums[];
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / P::size;

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  for (int row = static_cast<int>(blockIdx.x); row < rows; row += static_cast<int>(gridDim.x)) {
    const int row_pack_base = row * vec_count;
    float square_sum = 0.0f;

    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
      const int pack_idx = row_pack_base + vec_idx;
      float acc[P::size];
      const P first = reinterpret_cast<const P*>(data.ptrs[0])[pack_idx];
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        acc[i] = to_float(first.data[i]);
      }
#pragma unroll
      for (int r = 1; r < nranks; ++r) {
        const P peer = reinterpret_cast<const P*>(data.ptrs[r])[pack_idx];
#pragma unroll
        for (int i = 0; i < P::size; ++i) {
          acc[i] += to_float(peer.data[i]);
        }
      }

      const P residual_pack = reinterpret_cast<const P*>(residual)[pack_idx];
      P residual_out_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        // Match the raw-output ABI: the all-reduce result is first rounded to
        // the model dtype, then the residual is added and rounded again.
        const T reduced_value = from_float<T>(acc[i]);
        const float x =
            to_float(reduced_value) + to_float(residual_pack.data[i]);
        residual_out_pack.data[i] = from_float<T>(x);
        const float stored = to_float(residual_out_pack.data[i]);
        square_sum += stored * stored;
      }
      reinterpret_cast<P*>(residual_out)[pack_idx] = residual_out_pack;
    }

    square_sum = fused_ar_block_sum(square_sum, warp_sums);
    const float scale = fused_ar_fast_rsqrt(square_sum / static_cast<float>(hidden) + eps);

    for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
      const int pack_idx = row_pack_base + vec_idx;
      const int col = vec_idx * P::size;
      const P residual_out_pack = reinterpret_cast<const P*>(residual_out)[pack_idx];
      P out_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        const float x = to_float(residual_out_pack.data[i]);
        const float w = load_weight_scalar<WT>(weight, col + i);
        out_pack.data[i] = from_float<T>(x * scale * w);
      }
      reinterpret_cast<P*>(norm_out)[pack_idx] = out_pack;
    }
  }

  multi_rank_barrier<nranks, false>(sg, self_sg, rank);
}

template <typename T, typename WT>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) fused_ar_residual_rmsnorm_no_raw_local_tmp_kernel(
    RankSignals sg,
    const T* __restrict__ residual,
    const WT* __restrict__ weight,
    T* __restrict__ norm_out,
    T* __restrict__ residual_out,
    int rank,
    int rows,
    int hidden,
    float eps) {
  using P = typename packed_t<T>::P;
  static_assert(P::size == 8, "fused AR-residual-RMSNorm no-raw currently expects 8 values per 16B pack");
  extern __shared__ float warp_sums[];
  const int row = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / P::size;
  const int row_pack_base = row * vec_count;
  const P* tmps = get_tmp_buf<P>(sg.signals[rank]);
  float square_sum = 0.0f;

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
    const int pack_idx = row_pack_base + vec_idx;
    const P reduced_pack = tmps[pack_idx];
    const P residual_pack = reinterpret_cast<const P*>(residual)[pack_idx];
    P residual_out_pack;
#pragma unroll
    for (int i = 0; i < P::size; ++i) {
      const float x = to_float(reduced_pack.data[i]) + to_float(residual_pack.data[i]);
      residual_out_pack.data[i] = from_float<T>(x);
      const float stored = to_float(residual_out_pack.data[i]);
      square_sum += stored * stored;
    }
    reinterpret_cast<P*>(residual_out)[pack_idx] = residual_out_pack;
  }

  square_sum = fused_ar_block_sum(square_sum, warp_sums);
  const float scale = fused_ar_fast_rsqrt(square_sum / static_cast<float>(hidden) + eps);

  for (int vec_idx = tid; vec_idx < vec_count; vec_idx += static_cast<int>(blockDim.x)) {
    const int pack_idx = row_pack_base + vec_idx;
    const int col = vec_idx * P::size;
    const P residual_out_pack = reinterpret_cast<const P*>(residual_out)[pack_idx];
    P out_pack;
#pragma unroll
    for (int i = 0; i < P::size; ++i) {
      const float x = to_float(residual_out_pack.data[i]);
      const float w = load_weight_scalar<WT>(weight, col + i);
      out_pack.data[i] = from_float<T>(x * scale * w);
    }
    reinterpret_cast<P*>(norm_out)[pack_idx] = out_pack;
  }
}

template <typename T, typename WT, int nranks, bool HasResidual, bool WriteReduced>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1) fused_ar_rmsnorm_2shot_kernel(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ residual,
    const WT* __restrict__ weight,
    T* __restrict__ norm_out,
    T* __restrict__ residual_out,
    T* __restrict__ reduced,
    int rank,
    int rows,
    int hidden,
    float eps) {
  if (device_data != nullptr) {
    data = *device_data;
  }
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  static_assert(P::size == 8, "fused 2-shot AR-RMSNorm expects 8 values per 16B pack");

  extern __shared__ float warp_sums[];
  __shared__ float rank8_smem[nranks == 8 ? (kMaxThreadsPerBlock << 1) : 1];
  const int tid = static_cast<int>(threadIdx.x);
  const int vec_count = hidden / P::size;
  const int generic_part = vec_count / nranks;

  multi_rank_barrier<nranks, true>(sg, self_sg, rank);

  if constexpr ((nranks & (nranks - 1)) == 0) {
    // This is the native MUSA 2-shot mapping used by custom_all_reduce_2shot.
    // A 32-lane shuffle group reduces rank inputs; MP31 warp-squad alignment is
    // handled by the launch block size.
    constexpr int nranks_sft = (nranks >> 1) - (nranks >> 3);
    constexpr int coalesce_num = 8;
    constexpr int coalesce_sft = 3;
    constexpr int group_stride_sft = nranks_sft + coalesce_sft;
    constexpr int vlen = P::size;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int target_rank = (tid >> coalesce_sft) & (nranks - 1);
    const int group_id = tid >> group_stride_sft;
    const int coalesce_tid = tid & (coalesce_num - 1);
    const int idx_in_row =
        coalesce_tid + (rank << coalesce_sft) + (group_id << group_stride_sft);
    using Vec = int16_t __attribute__((vector_size(16)));
    const Vec* target_ptr = reinterpret_cast<const Vec*>(data.ptrs[target_rank]);
    Vec* self_tmp = get_tmp_buf<Vec>(sg.signals[rank]);

    for (int row = static_cast<int>(blockIdx.x); row < rows;
         row += static_cast<int>(gridDim.x)) {
      const int row_vec_base = row * vec_count;
      int idx_base = 0;
      do {
        const int vec_idx = idx_in_row + idx_base;
        float acc[vlen] = {0.0f};
        if (vec_idx < vec_count) {
#if SGL_CUSTOM_AR_VECTOR_LOAD
          const Vec raw = target_ptr[row_vec_base + vec_idx];
          const T* src = reinterpret_cast<const T*>(&raw);
#else
          const T* src =
              reinterpret_cast<const T*>(&target_ptr[row_vec_base + vec_idx]);
#endif
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            acc[i] = to_float(src[i]);
          }
        }
        shfl_reduce<T, nranks, vlen>(acc);
        if constexpr (nranks == 8) {
          if (lane < coalesce_num) {
#pragma unroll
            for (int i = 0; i < vlen; ++i) {
              rank8_smem[warp * vlen * coalesce_num + coalesce_tid * vlen + i] =
                  acc[i];
            }
          }
          __syncthreads_lm();
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            acc[i] +=
                rank8_smem[(warp ^ 1) * vlen * coalesce_num + coalesce_tid * vlen + i];
          }
        }
        if (rank == target_rank && vec_idx < vec_count) {
          Vec result;
#pragma unroll
          for (int i = 0; i < vlen; ++i) {
            reinterpret_cast<T*>(&result)[i] = downcast_s<T>(acc[i]);
          }
          self_tmp[row_vec_base + vec_idx] = result;
        }
        if constexpr (nranks == 8) {
          // Do not let the next iteration overwrite LDS while another squad is
          // still consuming the current rank-pair reduction.
          __syncthreads_lm();
        }
        idx_base += static_cast<int>(blockDim.x);
      } while (idx_base < vec_count);
    }
  } else {
    // TP6 keeps the generic MUSA two-stage reduction because the shuffle
    // target-rank mapping requires a power-of-two rank count.
    const int vec_start = rank * generic_part;
    const int vec_end = rank == nranks - 1 ? vec_count : vec_start + generic_part;
    const P* ptrs[nranks];
#pragma unroll
    for (int r = 0; r < nranks; ++r) {
      ptrs[r] = reinterpret_cast<const P*>(data.ptrs[r]);
    }
    P* self_tmp = get_tmp_buf<P>(sg.signals[rank]);
    for (int row = static_cast<int>(blockIdx.x); row < rows;
         row += static_cast<int>(gridDim.x)) {
      const int row_pack_base = row * vec_count;
      for (int vec_idx = vec_start + tid; vec_idx < vec_end;
           vec_idx += static_cast<int>(blockDim.x)) {
        const int pack_idx = row_pack_base + vec_idx;
        self_tmp[pack_idx] = packed_reduce<P, nranks, A>(ptrs, pack_idx);
      }
    }
  }

  __musa_barrier_slc();
  __syncthreads_lm();
  if (tid == 0) {
    __threadfence_system_noflush();
  }
  multi_rank_barrier<nranks, false, true>(sg, self_sg, rank);

  for (int row = static_cast<int>(blockIdx.x); row < rows;
       row += static_cast<int>(gridDim.x)) {
    const int row_pack_base = row * vec_count;
    float square_sum = 0.0f;

    for (int vec_idx = tid; vec_idx < vec_count;
         vec_idx += static_cast<int>(blockDim.x)) {
      const int pack_idx = row_pack_base + vec_idx;
      const int owner = fused_ar_2shot_owner<nranks>(vec_idx, generic_part);
      const P reduced_pack = get_tmp_buf<P>(sg.signals[owner])[pack_idx];
      P norm_input_pack = reduced_pack;
      if constexpr (WriteReduced) {
        reinterpret_cast<P*>(reduced)[pack_idx] = reduced_pack;
      }
      if constexpr (HasResidual) {
        const P residual_pack = reinterpret_cast<const P*>(residual)[pack_idx];
#pragma unroll
        for (int i = 0; i < P::size; ++i) {
          norm_input_pack.data[i] = downcast_s<T>(
              to_float(reduced_pack.data[i]) + to_float(residual_pack.data[i]));
        }
        reinterpret_cast<P*>(residual_out)[pack_idx] = norm_input_pack;
      }
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        const float x = to_float(norm_input_pack.data[i]);
        square_sum += x * x;
      }
    }

    square_sum = fused_ar_block_sum(square_sum, warp_sums);
    const float scale = fused_ar_fast_rsqrt(square_sum / static_cast<float>(hidden) + eps);

    for (int vec_idx = tid; vec_idx < vec_count;
         vec_idx += static_cast<int>(blockDim.x)) {
      const int pack_idx = row_pack_base + vec_idx;
      const int col = vec_idx * P::size;
      P norm_input_pack;
      if constexpr (HasResidual) {
        norm_input_pack = reinterpret_cast<const P*>(residual_out)[pack_idx];
      } else {
        norm_input_pack = reinterpret_cast<const P*>(reduced)[pack_idx];
      }
      P out_pack;
#pragma unroll
      for (int i = 0; i < P::size; ++i) {
        const float x = to_float(norm_input_pack.data[i]);
        const float w = load_weight_scalar<WT>(weight, col + i);
        out_pack.data[i] = from_float<T>(x * scale * w);
      }
      reinterpret_cast<P*>(norm_out)[pack_idx] = out_pack;
    }
  }
}

template <typename T, typename WT, int nranks>
void launch_fused_ar_rmsnorm(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const WT* weight,
    T* norm_out,
    T* reduced,
    int rank,
    int rows,
    int hidden,
    int shot,
    float eps,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  TVM_FFI_ICHECK_EQ(pack, 8);
  TVM_FFI_ICHECK_EQ(hidden % pack, 0);
  TVM_FFI_ICHECK_GT(rows, 0);
  TVM_FFI_ICHECK_GT(hidden, 0);
  TVM_FFI_ICHECK(shot == 1 || shot == 2) << "shot must be 1 or 2";
  const int64_t packed_size64 = static_cast<int64_t>(rows) * static_cast<int64_t>(hidden / pack);
  TVM_FFI_ICHECK_LE(packed_size64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  if (shot == 1) {
    if constexpr (nranks == 2) {
      if (launch_fused_ar_rmsnorm_tp2_specialized<T, WT, false, true>(
              data,
              device_data,
              sg,
              self_sg,
              static_cast<const T*>(nullptr),
              weight,
              norm_out,
              static_cast<T*>(nullptr),
              reduced,
              rank,
              rows,
              hidden,
              eps,
              stream)) {
        return;
      }
    }
  }
  const int rms_threads = fused_ar_vec8_block_threads(hidden);
  const int rms_smem = ((rms_threads + 31) / 32) * static_cast<int>(sizeof(float));
  const int two_shot_threads = fused_ar_vec8_2shot_block_threads(hidden);
  const int two_shot_smem = ((two_shot_threads + 31) / 32) * static_cast<int>(sizeof(float));

  if (shot == 1) {
    const int blocks = std::min(rows, kMaxBlocks);
    fused_ar_rmsnorm_1stage_kernel<T, WT, nranks><<<blocks, rms_threads, rms_smem, stream>>>(
        data, device_data, sg, self_sg, weight, norm_out, reduced, rank, rows, hidden, eps);
  } else if (shot == 2) {
    const int blocks = fused_ar_rmsnorm_2shot_blocks<nranks>(rows, hidden);
    fused_ar_rmsnorm_2shot_kernel<T, WT, nranks, false, true>
        <<<blocks, two_shot_threads, two_shot_smem, stream>>>(
            data,
            device_data,
            sg,
            self_sg,
            static_cast<const T*>(nullptr),
            weight,
            norm_out,
            static_cast<T*>(nullptr),
            reduced,
            rank,
            rows,
            hidden,
            eps);
  } else {
    TVM_FFI_THROW(ValueError) << "shot must be 1 or 2";
  }
}

template <typename T, typename WT>
void dispatch_fused_world_size(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const WT* weight,
    T* norm_out,
    T* reduced,
    int rank,
    int world_size,
    int rows,
    int hidden,
    int shot,
    float eps,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_fused_ar_rmsnorm<T, WT, 2>(data, device_data, sg, self_sg, weight, norm_out, reduced, rank, rows, hidden, shot, eps, stream);
      break;
    case 4:
      launch_fused_ar_rmsnorm<T, WT, 4>(data, device_data, sg, self_sg, weight, norm_out, reduced, rank, rows, hidden, shot, eps, stream);
      break;
    case 6:
      launch_fused_ar_rmsnorm<T, WT, 6>(data, device_data, sg, self_sg, weight, norm_out, reduced, rank, rows, hidden, shot, eps, stream);
      break;
    case 8:
      launch_fused_ar_rmsnorm<T, WT, 8>(data, device_data, sg, self_sg, weight, norm_out, reduced, rank, rows, hidden, shot, eps, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T, typename WT, int nranks>
void launch_fused_ar_residual_rmsnorm(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual,
    const WT* weight,
    T* norm_out,
    T* residual_out,
    T* reduced,
    int rank,
    int rows,
    int hidden,
    int shot,
    float eps,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  TVM_FFI_ICHECK_EQ(pack, 8);
  TVM_FFI_ICHECK_EQ(hidden % pack, 0);
  TVM_FFI_ICHECK_GT(rows, 0);
  TVM_FFI_ICHECK_GT(hidden, 0);
  TVM_FFI_ICHECK(shot == 1 || shot == 2) << "shot must be 1 or 2";
  const int64_t packed_size64 = static_cast<int64_t>(rows) * static_cast<int64_t>(hidden / pack);
  TVM_FFI_ICHECK_LE(packed_size64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  if (shot == 1) {
    if constexpr (nranks == 2) {
      if (launch_fused_ar_rmsnorm_tp2_specialized<T, WT, true, true>(
              data,
              device_data,
              sg,
              self_sg,
              residual,
              weight,
              norm_out,
              residual_out,
              reduced,
              rank,
              rows,
              hidden,
              eps,
              stream)) {
        return;
      }
    }
  }
  const int rms_threads = fused_ar_vec8_block_threads(hidden);
  const int rms_smem = ((rms_threads + 31) / 32) * static_cast<int>(sizeof(float));
  const int two_shot_threads = fused_ar_vec8_2shot_block_threads(hidden);
  const int two_shot_smem = ((two_shot_threads + 31) / 32) * static_cast<int>(sizeof(float));

  if (shot == 1) {
    const int blocks = std::min(rows, kMaxBlocks);
    fused_ar_residual_rmsnorm_1stage_kernel<T, WT, nranks><<<blocks, rms_threads, rms_smem, stream>>>(
        data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, reduced, rank, rows, hidden, eps);
  } else if (shot == 2) {
    const int blocks = fused_ar_rmsnorm_2shot_blocks<nranks>(rows, hidden);
    fused_ar_rmsnorm_2shot_kernel<T, WT, nranks, true, true>
        <<<blocks, two_shot_threads, two_shot_smem, stream>>>(
            data,
            device_data,
            sg,
            self_sg,
            residual,
            weight,
            norm_out,
            residual_out,
            reduced,
            rank,
            rows,
            hidden,
            eps);
  } else {
    TVM_FFI_THROW(ValueError) << "shot must be 1 or 2";
  }
}

template <typename T, typename WT>
void dispatch_fused_residual_world_size(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual,
    const WT* weight,
    T* norm_out,
    T* residual_out,
    T* reduced,
    int rank,
    int world_size,
    int rows,
    int hidden,
    int shot,
    float eps,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_fused_ar_residual_rmsnorm<T, WT, 2>(data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, reduced, rank, rows, hidden, shot, eps, stream);
      break;
    case 4:
      launch_fused_ar_residual_rmsnorm<T, WT, 4>(data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, reduced, rank, rows, hidden, shot, eps, stream);
      break;
    case 6:
      launch_fused_ar_residual_rmsnorm<T, WT, 6>(data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, reduced, rank, rows, hidden, shot, eps, stream);
      break;
    case 8:
      launch_fused_ar_residual_rmsnorm<T, WT, 8>(data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, reduced, rank, rows, hidden, shot, eps, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T, typename WT, int nranks>
void launch_fused_ar_residual_rmsnorm_no_raw(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual,
    const WT* weight,
    T* norm_out,
    T* residual_out,
    int rank,
    int rows,
    int hidden,
    int shot,
    float eps,
    musaStream_t stream) {
  constexpr int pack = packed_t<T>::P::size;
  TVM_FFI_ICHECK_EQ(pack, 8);
  TVM_FFI_ICHECK_EQ(hidden % pack, 0);
  TVM_FFI_ICHECK_GT(rows, 0);
  TVM_FFI_ICHECK_GT(hidden, 0);
  TVM_FFI_ICHECK(shot == 1 || shot == 2) << "shot must be 1 or 2";
  const int64_t packed_size64 = static_cast<int64_t>(rows) * static_cast<int64_t>(hidden / pack);
  TVM_FFI_ICHECK_LE(packed_size64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  if (shot == 1) {
    if constexpr (nranks == 2) {
      if (launch_fused_ar_rmsnorm_tp2_specialized<T, WT, true, false>(
              data,
              device_data,
              sg,
              self_sg,
              residual,
              weight,
              norm_out,
              residual_out,
              static_cast<T*>(nullptr),
              rank,
              rows,
              hidden,
              eps,
              stream)) {
        return;
      }
    }
  }
  const int rms_threads = fused_ar_vec8_block_threads(hidden);
  const int rms_smem = ((rms_threads + 31) / 32) * static_cast<int>(sizeof(float));
  const int two_shot_threads = fused_ar_vec8_2shot_block_threads(hidden);
  const int two_shot_smem = ((two_shot_threads + 31) / 32) * static_cast<int>(sizeof(float));

  if (shot == 1) {
    const int blocks = std::min(rows, kMaxBlocks);
    fused_ar_residual_rmsnorm_no_raw_1stage_kernel<T, WT, nranks><<<blocks, rms_threads, rms_smem, stream>>>(
        data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, rank, rows, hidden, eps);
  } else if (shot == 2) {
    const int blocks = fused_ar_rmsnorm_2shot_blocks<nranks>(rows, hidden);
    fused_ar_rmsnorm_2shot_kernel<T, WT, nranks, true, false>
        <<<blocks, two_shot_threads, two_shot_smem, stream>>>(
            data,
            device_data,
            sg,
            self_sg,
            residual,
            weight,
            norm_out,
            residual_out,
            static_cast<T*>(nullptr),
            rank,
            rows,
            hidden,
            eps);
  } else {
    TVM_FFI_THROW(ValueError) << "shot must be 1 or 2";
  }
}

template <typename T, typename WT>
void dispatch_fused_residual_no_raw_world_size(
    RankData data,
    const RankData* device_data,
    RankSignals sg,
    Signal* self_sg,
    const T* residual,
    const WT* weight,
    T* norm_out,
    T* residual_out,
    int rank,
    int world_size,
    int rows,
    int hidden,
    int shot,
    float eps,
    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_fused_ar_residual_rmsnorm_no_raw<T, WT, 2>(data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, rank, rows, hidden, shot, eps, stream);
      break;
    case 4:
      launch_fused_ar_residual_rmsnorm_no_raw<T, WT, 4>(data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, rank, rows, hidden, shot, eps, stream);
      break;
    case 6:
      launch_fused_ar_residual_rmsnorm_no_raw<T, WT, 6>(data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, rank, rows, hidden, shot, eps, stream);
      break;
    case 8:
      launch_fused_ar_residual_rmsnorm_no_raw<T, WT, 8>(data, device_data, sg, self_sg, residual, weight, norm_out, residual_out, rank, rows, hidden, shot, eps, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T, int nranks, bool indirect>
void launch_ar(RankData data, const RankData* data_ptr, RankSignals sg, Signal* self_sg, T* out, int rank, int size, int shot, musaStream_t stream) {
  const int pack = packed_t<T>::P::size;
  TVM_FFI_ICHECK_EQ(size % pack, 0);
  int packed_size = size / pack;
  int blocks = std::min(std::min(kDefaultBlockLimit, kMaxBlocks), (packed_size + kDefaultThreads - 1) / kDefaultThreads);
  if (blocks <= 0) {
    return;
  }
  if (shot == 1) {
    cross_device_reduce_1stage<T, nranks, indirect><<<blocks, kDefaultThreads, 0, stream>>>(data, data_ptr, sg, self_sg, out, rank, packed_size);
  } else if (shot == 2) {
    if constexpr (std::is_same<T, float>::value || nranks == 6) {
      cross_device_reduce_2stage<T, nranks, indirect><<<blocks, kDefaultThreads, 0, stream>>>(data, data_ptr, sg, self_sg, out, rank, packed_size);
    } else {
      custom_all_reduce_2shot<T, nranks, indirect><<<blocks, kDefaultThreads, 0, stream>>>(data, data_ptr, sg, self_sg, out, rank, packed_size);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "shot must be 1 or 2";
  }
}

template <typename T>
void dispatch_world_size(RankData data, RankSignals sg, Signal* self_sg, T* out, int rank, int world_size, int size, int shot, musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_ar<T, 2, false>(data, nullptr, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 4:
      launch_ar<T, 4, false>(data, nullptr, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 6:
      launch_ar<T, 6, false>(data, nullptr, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 8:
      launch_ar<T, 8, false>(data, nullptr, sg, self_sg, out, rank, size, shot, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename T>
void dispatch_world_size_indirect(const RankData* data_ptr, RankSignals sg, Signal* self_sg, T* out, int rank, int world_size, int size, int shot, musaStream_t stream) {
  RankData data{};
  switch (world_size) {
    case 2:
      launch_ar<T, 2, true>(data, data_ptr, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 4:
      launch_ar<T, 4, true>(data, data_ptr, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 6:
      launch_ar<T, 6, true>(data, data_ptr, sg, self_sg, out, rank, size, shot, stream);
      break;
    case 8:
      launch_ar<T, 8, true>(data, data_ptr, sg, self_sg, out, rank, size, shot, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "world_size must be one of 2/4/6/8";
  }
}

template <typename InputT, typename OutputT>
void dispatch_all_gather_world_size(RankData data, RankSignals sg,
                                    Signal* self_sg, OutputT* out, int rank,
                                    int world_size, int rows, int shard_size,
                                    musaStream_t stream) {
  switch (world_size) {
    case 2:
      launch_all_gather_last_dim<InputT, OutputT, 2>(
          data, sg, self_sg, out, rank, rows, shard_size, stream);
      break;
    case 4:
      launch_all_gather_last_dim<InputT, OutputT, 4>(
          data, sg, self_sg, out, rank, rows, shard_size, stream);
      break;
    case 8:
      launch_all_gather_last_dim<InputT, OutputT, 8>(
          data, sg, self_sg, out, rank, rows, shard_size, stream);
      break;
    default:
      TVM_FFI_THROW(ValueError) << "all-gather world_size must be one of 2/4/8";
  }
}
inline void validate_world_size(int64_t world_size) {
  TVM_FFI_ICHECK(
      world_size == 2 || world_size == 4 || world_size == 6 ||
      world_size == 8)
      << "world_size must be one of 2/4/6/8";
  TVM_FFI_ICHECK_LE(world_size, kMaxRanks);
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

void vllm_musa_custom_ar_launch_all_gather(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size) {
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
  TVM_FFI_ICHECK_EQ(inp.ndim(), 2);
  TVM_FFI_ICHECK_EQ(out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(inp.size(0), out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1) * world_size, out.size(1));
  const bool same_dtype = dtype_equal(inp.dtype(), out.dtype());
  const bool bfloat16_to_float32 =
      dtype_equal(inp.dtype(), dl_bfloat16) &&
      dtype_equal(out.dtype(), dl_float32);
  TVM_FFI_ICHECK(same_dtype || bfloat16_to_float32);

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
  // The local input is still live for this eager call and is typically hot
  // from LM-head GEMM. Peers read the published IPC staging copy, while this
  // rank can read its original shard directly.
  data.ptrs[rank] = inp.data_ptr();

  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(out.device());
  const int64_t input_numel = tensor_numel(inp);
  const int64_t output_numel = tensor_numel(out);
  TVM_FFI_ICHECK_LE(input_numel,
                    static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  TVM_FFI_ICHECK_LE(output_numel,
                    static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int64_t input_element_bytes =
      (static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8;
  const int64_t output_element_bytes =
      (static_cast<int64_t>(out.dtype().bits) * out.dtype().lanes + 7) / 8;
  const int64_t input_nbytes = input_numel * input_element_bytes;
  const int64_t output_nbytes = output_numel * output_element_bytes;
  TVM_FFI_ICHECK_LE(input_nbytes, max_size_bytes);
  TVM_FFI_ICHECK_LE(output_nbytes, max_size_bytes);
  const musaError_t copy_err = musaMemcpyAsync(
      reinterpret_cast<void*>(self_buffer_ptr), inp.data_ptr(),
      static_cast<size_t>(input_nbytes), musaMemcpyDeviceToDevice, stream);
  TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
      << "MUSA custom all-gather copy failed: "
      << musaGetErrorString(copy_err);

  const int rows = static_cast<int>(inp.size(0));
  const int shard_size = static_cast<int>(inp.size(1));
  if (bfloat16_to_float32) {
    dispatch_all_gather_world_size<__mt_bfloat16, float>(
        data, sg, self_sg, static_cast<float*>(out.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), rows, shard_size,
        stream);
  } else if (dtype_equal(out.dtype(), dl_float16)) {
    dispatch_all_gather_world_size<half, half>(
        data, sg, self_sg, static_cast<half*>(out.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), rows, shard_size,
        stream);
  } else if (dtype_equal(out.dtype(), dl_bfloat16)) {
    dispatch_all_gather_world_size<__mt_bfloat16, __mt_bfloat16>(
        data, sg, self_sg, static_cast<__mt_bfloat16*>(out.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), rows, shard_size,
        stream);
  } else if (dtype_equal(out.dtype(), dl_float32)) {
    dispatch_all_gather_world_size<float, float>(
        data, sg, self_sg, static_cast<float*>(out.data_ptr()),
        static_cast<int>(rank), static_cast<int>(world_size), rows, shard_size,
        stream);
  } else {
    TVM_FFI_THROW(ValueError)
        << "custom all-gather only supports same-dtype fp16/bf16/fp32 or "
           "bf16-to-fp32";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess)
      << "MUSA custom all-gather kernel failed: " << musaGetErrorString(err);
}

void vllm_musa_custom_ar_launch_registered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
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
  const auto* signal_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(signal_ptrs[i]);
  }
  RankData data{};
  const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
  for (int i = 0; i < kMaxRanks; ++i) {
    data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
  }
  TVM_FFI_ICHECK_EQ(data.ptrs[rank], inp.data_ptr());

  auto* self_sg = sg.signals[rank];
  auto stream = get_stream(out.device());
  const int64_t numel64 = tensor_numel(out);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
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

void vllm_musa_custom_ar_launch_graph_registered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView out,
    int64_t rank,
    int64_t world_size,
    int64_t shot) {
  CHECK_MUSA_CONTIGUOUS(rank_data);
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(out);
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(tensor_numel(rank_data), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), out.dtype()));
  TVM_FFI_ICHECK_EQ(tensor_numel(inp), tensor_numel(out));

  RankSignals sg{};
  const auto* signal_ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(signal_ptrs[i]);
  }
  const auto* data_ptr = static_cast<const RankData*>(rank_data.data_ptr());
  auto* self_sg = sg.signals[rank];
  auto stream = get_stream(out.device());
  const int64_t numel64 = tensor_numel(out);
  TVM_FFI_ICHECK_LE(numel64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  const int size = static_cast<int>(numel64);

  if (dtype_equal(out.dtype(), dl_float16)) {
    dispatch_world_size_indirect(data_ptr, sg, self_sg, static_cast<half*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_bfloat16)) {
    dispatch_world_size_indirect(data_ptr, sg, self_sg, static_cast<__mt_bfloat16*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else if (dtype_equal(out.dtype(), dl_float32)) {
    dispatch_world_size_indirect(data_ptr, sg, self_sg, static_cast<float*>(out.data_ptr()), static_cast<int>(rank), static_cast<int>(world_size), size, static_cast<int>(shot), stream);
  } else {
    TVM_FFI_THROW(ValueError) << "custom ar only supports fp16/bf16/fp32";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA custom AR kernel failed: " << musaGetErrorString(err);
}

void vllm_musa_fused_ar_rmsnorm_launch_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView weight,
    ffi::TensorView norm_out,
    ffi::TensorView reduced,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t shot,
    double eps) {
  validate_world_size(world_size);
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(weight);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(reduced);
  const bool rank_data_on_cpu = rank_data.device().device_type == kDLCPU;
  const bool rank_data_on_musa =
      rank_data.device().device_type == inp.device().device_type;
  TVM_FFI_ICHECK(rank_data_on_cpu || rank_data_on_musa);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK_EQ(inp.ndim(), 2);
  TVM_FFI_ICHECK_EQ(norm_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(reduced.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inp.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), reduced.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), reduced.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(1), weight.size(0));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), reduced.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), weight.dtype()) || dtype_equal(weight.dtype(), dl_float32));
  TVM_FFI_ICHECK_EQ(inp.device().device_id, weight.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, norm_out.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, reduced.device().device_id);

  const int64_t rows64 = inp.size(0);
  const int64_t hidden64 = inp.size(1);
  TVM_FFI_ICHECK_GT(rows64, 0);
  TVM_FFI_ICHECK_GT(hidden64, 0);
  TVM_FFI_ICHECK_EQ(hidden64 % 8, 0);
  TVM_FFI_ICHECK_LE(rows64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  TVM_FFI_ICHECK_LE(hidden64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  TVM_FFI_ICHECK_LE(rows64 * hidden64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  RankSignals sg{};
  const auto* ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(ptrs[i]);
  }
  RankData data{};
  const RankData* device_data = nullptr;
  if (rank_data_on_cpu) {
    const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
    for (int i = 0; i < kMaxRanks; ++i) {
      data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
    }
  } else {
    TVM_FFI_ICHECK_EQ(rank_data.device().device_id, inp.device().device_id);
    device_data = reinterpret_cast<const RankData*>(rank_data.data_ptr());
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(norm_out.device());
  const int64_t numel64 = tensor_numel(inp);
  const int64_t nbytes = numel64 * ((static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);
  if (rank_data_on_cpu) {
    const musaError_t copy_err = musaMemcpyAsync(
        reinterpret_cast<void*>(self_buffer_ptr),
        inp.data_ptr(),
        static_cast<size_t>(nbytes),
        musaMemcpyDeviceToDevice,
        stream);
    TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
        << "MUSA fused AR-RMSNorm input copy failed: "
        << musaGetErrorString(copy_err);
  }

  const int rows = static_cast<int>(rows64);
  const int hidden = static_cast<int>(hidden64);
  if (dtype_equal(inp.dtype(), dl_float16)) {
    if (dtype_equal(weight.dtype(), dl_float32)) {
      dispatch_fused_world_size<half, float>(
          data, device_data, sg, self_sg, static_cast<const float*>(weight.data_ptr()),
          static_cast<half*>(norm_out.data_ptr()), static_cast<half*>(reduced.data_ptr()),
          static_cast<int>(rank), static_cast<int>(world_size), rows, hidden,
          static_cast<int>(shot), static_cast<float>(eps), stream);
    } else {
      dispatch_fused_world_size<half, half>(
          data, device_data, sg, self_sg, static_cast<const half*>(weight.data_ptr()),
          static_cast<half*>(norm_out.data_ptr()), static_cast<half*>(reduced.data_ptr()),
          static_cast<int>(rank), static_cast<int>(world_size), rows, hidden,
          static_cast<int>(shot), static_cast<float>(eps), stream);
    }
  } else if (dtype_equal(inp.dtype(), dl_bfloat16)) {
    if (dtype_equal(weight.dtype(), dl_float32)) {
      dispatch_fused_world_size<__mt_bfloat16, float>(
          data, device_data, sg, self_sg, static_cast<const float*>(weight.data_ptr()),
          static_cast<__mt_bfloat16*>(norm_out.data_ptr()), static_cast<__mt_bfloat16*>(reduced.data_ptr()),
          static_cast<int>(rank), static_cast<int>(world_size), rows, hidden,
          static_cast<int>(shot), static_cast<float>(eps), stream);
    } else {
      dispatch_fused_world_size<__mt_bfloat16, __mt_bfloat16>(
          data, device_data, sg, self_sg, static_cast<const __mt_bfloat16*>(weight.data_ptr()),
          static_cast<__mt_bfloat16*>(norm_out.data_ptr()), static_cast<__mt_bfloat16*>(reduced.data_ptr()),
          static_cast<int>(rank), static_cast<int>(world_size), rows, hidden,
          static_cast<int>(shot), static_cast<float>(eps), stream);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "fused AR-RMSNorm only supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA fused AR-RMSNorm kernel failed: " << musaGetErrorString(err);
}

void vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView residual,
    ffi::TensorView weight,
    ffi::TensorView norm_out,
    ffi::TensorView residual_out,
    ffi::TensorView reduced,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t shot,
    double eps) {
  validate_world_size(world_size);
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(residual);
  CHECK_MUSA_CONTIGUOUS(weight);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  CHECK_MUSA_CONTIGUOUS(reduced);
  const bool rank_data_on_cpu = rank_data.device().device_type == kDLCPU;
  const bool rank_data_on_musa =
      rank_data.device().device_type == inp.device().device_type;
  TVM_FFI_ICHECK(rank_data_on_cpu || rank_data_on_musa);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK_EQ(inp.ndim(), 2);
  TVM_FFI_ICHECK_EQ(residual.ndim(), 2);
  TVM_FFI_ICHECK_EQ(norm_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(residual_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(reduced.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inp.size(0), residual.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), reduced.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), reduced.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(1), weight.size(0));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), reduced.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), weight.dtype()) || dtype_equal(weight.dtype(), dl_float32));
  TVM_FFI_ICHECK_EQ(inp.device().device_id, residual.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, weight.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, norm_out.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, residual_out.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, reduced.device().device_id);

  const int64_t rows64 = inp.size(0);
  const int64_t hidden64 = inp.size(1);
  TVM_FFI_ICHECK_GT(rows64, 0);
  TVM_FFI_ICHECK_GT(hidden64, 0);
  TVM_FFI_ICHECK_EQ(hidden64 % 8, 0);
  TVM_FFI_ICHECK_LE(rows64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  TVM_FFI_ICHECK_LE(hidden64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  TVM_FFI_ICHECK_LE(rows64 * hidden64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  RankSignals sg{};
  const auto* ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(ptrs[i]);
  }
  RankData data{};
  const RankData* device_data = nullptr;
  if (rank_data_on_cpu) {
    const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
    for (int i = 0; i < kMaxRanks; ++i) {
      data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
    }
  } else {
    TVM_FFI_ICHECK_EQ(rank_data.device().device_id, inp.device().device_id);
    device_data = reinterpret_cast<const RankData*>(rank_data.data_ptr());
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(norm_out.device());
  const int64_t numel64 = tensor_numel(inp);
  const int64_t nbytes = numel64 * ((static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);
  if (rank_data_on_cpu) {
    const musaError_t copy_err = musaMemcpyAsync(
        reinterpret_cast<void*>(self_buffer_ptr),
        inp.data_ptr(),
        static_cast<size_t>(nbytes),
        musaMemcpyDeviceToDevice,
        stream);
    TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
        << "MUSA fused AR-residual-RMSNorm input copy failed: "
        << musaGetErrorString(copy_err);
  }

  const int rows = static_cast<int>(rows64);
  const int hidden = static_cast<int>(hidden64);
  if (dtype_equal(inp.dtype(), dl_float16)) {
    if (dtype_equal(weight.dtype(), dl_float32)) {
      dispatch_fused_residual_world_size<half, float>(
          data, device_data, sg, self_sg, static_cast<const half*>(residual.data_ptr()),
          static_cast<const float*>(weight.data_ptr()), static_cast<half*>(norm_out.data_ptr()),
          static_cast<half*>(residual_out.data_ptr()), static_cast<half*>(reduced.data_ptr()),
          static_cast<int>(rank),
          static_cast<int>(world_size), rows, hidden, static_cast<int>(shot),
          static_cast<float>(eps), stream);
    } else {
      dispatch_fused_residual_world_size<half, half>(
          data, device_data, sg, self_sg, static_cast<const half*>(residual.data_ptr()),
          static_cast<const half*>(weight.data_ptr()), static_cast<half*>(norm_out.data_ptr()),
          static_cast<half*>(residual_out.data_ptr()), static_cast<half*>(reduced.data_ptr()),
          static_cast<int>(rank),
          static_cast<int>(world_size), rows, hidden, static_cast<int>(shot),
          static_cast<float>(eps), stream);
    }
  } else if (dtype_equal(inp.dtype(), dl_bfloat16)) {
    if (dtype_equal(weight.dtype(), dl_float32)) {
      dispatch_fused_residual_world_size<__mt_bfloat16, float>(
          data, device_data, sg, self_sg, static_cast<const __mt_bfloat16*>(residual.data_ptr()),
          static_cast<const float*>(weight.data_ptr()), static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
          static_cast<__mt_bfloat16*>(residual_out.data_ptr()), static_cast<__mt_bfloat16*>(reduced.data_ptr()),
          static_cast<int>(rank),
          static_cast<int>(world_size), rows, hidden, static_cast<int>(shot),
          static_cast<float>(eps), stream);
    } else {
      dispatch_fused_residual_world_size<__mt_bfloat16, __mt_bfloat16>(
          data, device_data, sg, self_sg, static_cast<const __mt_bfloat16*>(residual.data_ptr()),
          static_cast<const __mt_bfloat16*>(weight.data_ptr()), static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
          static_cast<__mt_bfloat16*>(residual_out.data_ptr()), static_cast<__mt_bfloat16*>(reduced.data_ptr()),
          static_cast<int>(rank),
          static_cast<int>(world_size), rows, hidden, static_cast<int>(shot),
          static_cast<float>(eps), stream);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "fused AR-residual-RMSNorm only supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA fused AR-residual-RMSNorm kernel failed: " << musaGetErrorString(err);
}

void vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered(
    ffi::TensorView rank_data,
    ffi::TensorView signal_ptrs_cpu,
    ffi::TensorView inp,
    ffi::TensorView residual,
    ffi::TensorView weight,
    ffi::TensorView norm_out,
    ffi::TensorView residual_out,
    int64_t self_signal_ptr,
    int64_t self_buffer_ptr,
    int64_t max_size_bytes,
    int64_t rank,
    int64_t world_size,
    int64_t shot,
    double eps) {
  validate_world_size(world_size);
  CHECK_MUSA_CONTIGUOUS(inp);
  CHECK_MUSA_CONTIGUOUS(residual);
  CHECK_MUSA_CONTIGUOUS(weight);
  CHECK_MUSA_CONTIGUOUS(norm_out);
  CHECK_MUSA_CONTIGUOUS(residual_out);
  const bool rank_data_on_cpu = rank_data.device().device_type == kDLCPU;
  const bool rank_data_on_musa =
      rank_data.device().device_type == inp.device().device_type;
  TVM_FFI_ICHECK(rank_data_on_cpu || rank_data_on_musa);
  TVM_FFI_ICHECK(rank_data.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(rank_data.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(rank_data.size(0), kMaxRanks);
  TVM_FFI_ICHECK_EQ(signal_ptrs_cpu.device().device_type, kDLCPU);
  TVM_FFI_ICHECK(signal_ptrs_cpu.IsContiguous());
  TVM_FFI_ICHECK(dtype_equal(signal_ptrs_cpu.dtype(), dl_int64));
  TVM_FFI_ICHECK_GE(signal_ptrs_cpu.size(0), world_size);
  TVM_FFI_ICHECK(rank >= 0 && rank < world_size);
  TVM_FFI_ICHECK_EQ(inp.ndim(), 2);
  TVM_FFI_ICHECK_EQ(residual.ndim(), 2);
  TVM_FFI_ICHECK_EQ(norm_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(residual_out.ndim(), 2);
  TVM_FFI_ICHECK_EQ(weight.ndim(), 1);
  TVM_FFI_ICHECK_EQ(inp.size(0), residual.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), norm_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), norm_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(0), residual_out.size(0));
  TVM_FFI_ICHECK_EQ(inp.size(1), residual_out.size(1));
  TVM_FFI_ICHECK_EQ(inp.size(1), weight.size(0));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), norm_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), residual_out.dtype()));
  TVM_FFI_ICHECK(dtype_equal(inp.dtype(), weight.dtype()) || dtype_equal(weight.dtype(), dl_float32));
  TVM_FFI_ICHECK_EQ(inp.device().device_id, residual.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, weight.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, norm_out.device().device_id);
  TVM_FFI_ICHECK_EQ(inp.device().device_id, residual_out.device().device_id);

  const int64_t rows64 = inp.size(0);
  const int64_t hidden64 = inp.size(1);
  TVM_FFI_ICHECK_GT(rows64, 0);
  TVM_FFI_ICHECK_GT(hidden64, 0);
  TVM_FFI_ICHECK_EQ(hidden64 % 8, 0);
  TVM_FFI_ICHECK_LE(rows64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  TVM_FFI_ICHECK_LE(hidden64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  TVM_FFI_ICHECK_LE(rows64 * hidden64, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  RankSignals sg{};
  const auto* ptrs = static_cast<const int64_t*>(signal_ptrs_cpu.data_ptr());
  for (int i = 0; i < world_size; ++i) {
    sg.signals[i] = reinterpret_cast<Signal*>(ptrs[i]);
  }
  RankData data{};
  const RankData* device_data = nullptr;
  if (rank_data_on_cpu) {
    const auto* rank_ptrs = static_cast<const int64_t*>(rank_data.data_ptr());
    for (int i = 0; i < kMaxRanks; ++i) {
      data.ptrs[i] = reinterpret_cast<const void*>(rank_ptrs[i]);
    }
  } else {
    TVM_FFI_ICHECK_EQ(rank_data.device().device_id, inp.device().device_id);
    device_data = reinterpret_cast<const RankData*>(rank_data.data_ptr());
  }
  auto* self_sg = reinterpret_cast<Signal*>(self_signal_ptr);
  auto stream = get_stream(norm_out.device());
  const int64_t numel64 = tensor_numel(inp);
  const int64_t nbytes = numel64 * ((static_cast<int64_t>(inp.dtype().bits) * inp.dtype().lanes + 7) / 8);
  TVM_FFI_ICHECK_LE(nbytes, max_size_bytes);
  if (rank_data_on_cpu) {
    const musaError_t copy_err = musaMemcpyAsync(
        reinterpret_cast<void*>(self_buffer_ptr),
        inp.data_ptr(),
        static_cast<size_t>(nbytes),
        musaMemcpyDeviceToDevice,
        stream);
    TVM_FFI_ICHECK_EQ(copy_err, musaSuccess)
        << "MUSA fused AR-residual-RMSNorm no-raw input copy failed: "
        << musaGetErrorString(copy_err);
  }

  const int rows = static_cast<int>(rows64);
  const int hidden = static_cast<int>(hidden64);
  if (dtype_equal(inp.dtype(), dl_float16)) {
    if (dtype_equal(weight.dtype(), dl_float32)) {
      dispatch_fused_residual_no_raw_world_size<half, float>(
          data, device_data, sg, self_sg, static_cast<const half*>(residual.data_ptr()),
          static_cast<const float*>(weight.data_ptr()), static_cast<half*>(norm_out.data_ptr()),
          static_cast<half*>(residual_out.data_ptr()), static_cast<int>(rank),
          static_cast<int>(world_size), rows, hidden, static_cast<int>(shot),
          static_cast<float>(eps), stream);
    } else {
      dispatch_fused_residual_no_raw_world_size<half, half>(
          data, device_data, sg, self_sg, static_cast<const half*>(residual.data_ptr()),
          static_cast<const half*>(weight.data_ptr()), static_cast<half*>(norm_out.data_ptr()),
          static_cast<half*>(residual_out.data_ptr()), static_cast<int>(rank),
          static_cast<int>(world_size), rows, hidden, static_cast<int>(shot),
          static_cast<float>(eps), stream);
    }
  } else if (dtype_equal(inp.dtype(), dl_bfloat16)) {
    if (dtype_equal(weight.dtype(), dl_float32)) {
      dispatch_fused_residual_no_raw_world_size<__mt_bfloat16, float>(
          data, device_data, sg, self_sg, static_cast<const __mt_bfloat16*>(residual.data_ptr()),
          static_cast<const float*>(weight.data_ptr()), static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
          static_cast<__mt_bfloat16*>(residual_out.data_ptr()), static_cast<int>(rank),
          static_cast<int>(world_size), rows, hidden, static_cast<int>(shot),
          static_cast<float>(eps), stream);
    } else {
      dispatch_fused_residual_no_raw_world_size<__mt_bfloat16, __mt_bfloat16>(
          data, device_data, sg, self_sg, static_cast<const __mt_bfloat16*>(residual.data_ptr()),
          static_cast<const __mt_bfloat16*>(weight.data_ptr()), static_cast<__mt_bfloat16*>(norm_out.data_ptr()),
          static_cast<__mt_bfloat16*>(residual_out.data_ptr()), static_cast<int>(rank),
          static_cast<int>(world_size), rows, hidden, static_cast<int>(shot),
          static_cast<float>(eps), stream);
    }
  } else {
    TVM_FFI_THROW(ValueError) << "fused AR-residual-RMSNorm no-raw only supports fp16/bf16";
  }
  const musaError_t err = musaGetLastError();
  TVM_FFI_ICHECK_EQ(err, musaSuccess) << "MUSA fused AR-residual-RMSNorm no-raw kernel failed: " << musaGetErrorString(err);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_custom_ar_meta_size, vllm_musa_custom_ar_meta_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_custom_ar_launch_unregistered, vllm_musa_custom_ar_launch_unregistered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_custom_ar_launch_all_gather,
                              vllm_musa_custom_ar_launch_all_gather);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_custom_ar_launch_registered, vllm_musa_custom_ar_launch_registered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_custom_ar_launch_graph_registered, vllm_musa_custom_ar_launch_graph_registered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_fused_ar_rmsnorm_launch_unregistered, vllm_musa_fused_ar_rmsnorm_launch_unregistered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered, vllm_musa_fused_ar_residual_rmsnorm_launch_unregistered);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered, vllm_musa_fused_ar_residual_rmsnorm_no_raw_launch_unregistered);
