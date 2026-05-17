#include <cmath>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int64_t kNopeDim = 448;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kHeadDim = kNopeDim + kRopeDim;
constexpr int64_t kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr int64_t kTokenScaleBytes = 8;
constexpr int64_t kQuantBlockSize = 64;
constexpr int kGroupedHeads = 4;
constexpr int kGroupedTileSlots = 16;
constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;
constexpr int kSinkFloat32 = 1;
constexpr int kSinkBFloat16 = 2;

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t value) {
  const int sign = (value & 0x80) ? -1 : 1;
  const int exponent = (value >> 3) & 0x0f;
  const int mantissa = value & 0x07;
  if (exponent == 0 && mantissa == 0) {
    return sign < 0 ? -0.0f : 0.0f;
  }
  if (exponent == 0) {
    return sign * ldexpf(static_cast<float>(mantissa), -9);
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return NAN;
  }
  return sign * ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f,
                       exponent - 7);
}

__device__ __forceinline__ float bf16_bytes_to_float(const uint8_t* ptr) {
  const uint32_t bits =
      (static_cast<uint32_t>(ptr[1]) << 24) |
      (static_cast<uint32_t>(ptr[0]) << 16);
  return __uint_as_float(bits);
}

__device__ __forceinline__ int64_t load_index(const void* ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t*>(ptr)[idx]);
  }
  return static_cast<int64_t>(static_cast<const int64_t*>(ptr)[idx]);
}

__device__ __forceinline__ float load_sink(const void* ptr, int kind,
                                           int64_t idx) {
  if (kind == kSinkBFloat16) {
    return __bfloat162float(static_cast<const __mt_bfloat16*>(ptr)[idx]);
  }
  return static_cast<const float*>(ptr)[idx];
}

__device__ __forceinline__ float logaddexpf_stable(float a, float b) {
  const float m = fmaxf(a, b);
  if (isinf(m)) {
    return m;
  }
  return m + logf(expf(a - m) + expf(b - m));
}

__device__ __forceinline__ float load_cache_value(
    const uint8_t* cache, int64_t raw_idx, int64_t dim, int64_t num_blocks,
    int64_t block_size, int64_t block_stride) {
  const int64_t block_idx = raw_idx / block_size;
  const int64_t pos_in_block = raw_idx - block_idx * block_size;
  const uint8_t* block_ptr = cache + block_idx * block_stride;
  const int64_t token_base = pos_in_block * kTokenDataBytes;
  if (dim < kNopeDim) {
    const int64_t scale_base =
        block_size * kTokenDataBytes + pos_in_block * kTokenScaleBytes;
    const uint8_t fp8 = block_ptr[token_base + dim];
    const uint8_t scale_byte = block_ptr[scale_base + dim / kQuantBlockSize];
    return fp8_e4m3fn_to_float(fp8) *
           ldexpf(1.0f, static_cast<int>(scale_byte) - 127);
  }
  const int64_t rope_dim = dim - kNopeDim;
  const uint8_t* bf16_ptr = block_ptr + token_base + kNopeDim + rope_dim * 2;
  return bf16_bytes_to_float(bf16_ptr);
}

__device__ __forceinline__ bool slot_info(
    int64_t slot, int64_t query, int64_t topk, int64_t extra_topk,
    int64_t num_kv_tokens, const void* indices, int index_kind,
    const void* lengths, int length_kind, const void* extra_indices,
    const void* extra_lengths, int64_t& raw_idx) {
  const bool is_extra = slot >= topk;
  const int64_t local_slot = is_extra ? slot - topk : slot;
  const int64_t local_topk = is_extra ? extra_topk : topk;
  const void* local_indices = is_extra ? extra_indices : indices;
  const void* local_lengths = is_extra ? extra_lengths : lengths;
  if (local_indices == nullptr || local_slot >= local_topk) {
    raw_idx = 0;
    return false;
  }
  raw_idx = load_index(local_indices, index_kind, query * local_topk + local_slot);
  const int64_t length = local_lengths == nullptr
                             ? local_topk
                             : load_index(local_lengths, length_kind, query);
  return raw_idx >= 0 && raw_idx < num_kv_tokens && local_slot < length;
}

__global__ void fp8_ds_mla_sparse_attention_kernel(
    const __mt_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache,
    const void* __restrict__ indices, const void* __restrict__ lengths,
    const void* __restrict__ attn_sink, const uint8_t* __restrict__ extra_cache,
    const void* __restrict__ extra_indices,
    const void* __restrict__ extra_lengths, __mt_bfloat16* __restrict__ output,
    float* __restrict__ lse, int index_kind, int length_kind, int sink_kind,
    int64_t num_queries, int64_t seq_len, int64_t num_heads, int64_t q_dim,
    int64_t topk, int64_t extra_topk, int64_t num_blocks, int64_t block_size,
    int64_t block_stride, int64_t extra_num_blocks, int64_t extra_block_size,
    int64_t extra_block_stride, float softmax_scale) {
  const int64_t query = static_cast<int64_t>(blockIdx.x);
  const int64_t head = static_cast<int64_t>(blockIdx.y);
  if (query >= num_queries || head >= num_heads) {
    return;
  }

  extern __shared__ float shared[];
  float* logits = shared;
  float* reduce = shared + topk + extra_topk;
  const int64_t total_topk = topk + extra_topk;
  const int64_t num_kv_tokens = num_blocks * block_size;
  const int64_t extra_num_kv_tokens = extra_num_blocks * extra_block_size;

  float max_logit = -INFINITY;
  int has_valid = 0;
  for (int64_t slot = 0; slot < total_topk; ++slot) {
    int64_t raw_idx = 0;
    const bool is_extra = slot >= topk;
    const bool valid =
        is_extra
            ? slot_info(slot, query, topk, extra_topk, extra_num_kv_tokens,
                        indices, index_kind, lengths, length_kind,
                        extra_indices, extra_lengths, raw_idx)
            : slot_info(slot, query, topk, extra_topk, num_kv_tokens, indices,
                        index_kind, lengths, length_kind, extra_indices,
                        extra_lengths, raw_idx);
    float partial = 0.0f;
    if (valid) {
      const uint8_t* local_cache = is_extra ? extra_cache : cache;
      const int64_t local_blocks = is_extra ? extra_num_blocks : num_blocks;
      const int64_t local_block_size = is_extra ? extra_block_size : block_size;
      const int64_t local_stride = is_extra ? extra_block_stride : block_stride;
      for (int64_t dim = threadIdx.x; dim < q_dim; dim += blockDim.x) {
        const int64_t q_offset = (query * num_heads + head) * q_dim + dim;
        const float q_value = __bfloat162float(q[q_offset]);
        partial += q_value *
                   load_cache_value(local_cache, raw_idx, dim, local_blocks,
                                    local_block_size, local_stride);
      }
    }
    reduce[threadIdx.x] = partial;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      const float logit = valid ? reduce[0] * softmax_scale : -INFINITY;
      logits[slot] = logit;
      if (valid) {
        has_valid = 1;
        max_logit = fmaxf(max_logit, logit);
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float key_lse = -INFINITY;
    float lse_for_output = -INFINITY;
    if (has_valid) {
      float sum_exp = 0.0f;
      for (int64_t slot = 0; slot < total_topk; ++slot) {
        if (!isinf(logits[slot])) {
          sum_exp += expf(logits[slot] - max_logit);
        }
      }
      key_lse = max_logit + logf(sum_exp);
      lse_for_output = key_lse;
    }
    if (attn_sink != nullptr) {
      const float sink = load_sink(attn_sink, sink_kind, head);
      lse_for_output = has_valid ? logaddexpf_stable(key_lse, sink) : sink;
    }
    reduce[0] = lse_for_output;
    reduce[1] = static_cast<float>(has_valid);
    const int64_t batch = query / seq_len;
    const int64_t seq = query - batch * seq_len;
    lse[(batch * num_heads + head) * seq_len + seq] =
        has_valid ? key_lse : INFINITY;
  }
  __syncthreads();

  const float lse_for_output = reduce[0];
  const bool output_has_valid = reduce[1] > 0.5f;
  for (int64_t dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
    float accum = 0.0f;
    if (output_has_valid) {
      for (int64_t slot = 0; slot < total_topk; ++slot) {
        if (isinf(logits[slot])) {
          continue;
        }
        int64_t raw_idx = 0;
        const bool is_extra = slot >= topk;
        const bool valid =
            is_extra
                ? slot_info(slot, query, topk, extra_topk, extra_num_kv_tokens,
                            indices, index_kind, lengths, length_kind,
                            extra_indices, extra_lengths, raw_idx)
                : slot_info(slot, query, topk, extra_topk, num_kv_tokens,
                            indices, index_kind, lengths, length_kind,
                            extra_indices, extra_lengths, raw_idx);
        if (!valid) {
          continue;
        }
        const uint8_t* local_cache = is_extra ? extra_cache : cache;
        const int64_t local_blocks = is_extra ? extra_num_blocks : num_blocks;
        const int64_t local_block_size = is_extra ? extra_block_size : block_size;
        const int64_t local_stride = is_extra ? extra_block_stride : block_stride;
        const float weight = expf(logits[slot] - lse_for_output);
        accum += weight * load_cache_value(local_cache, raw_idx, dim, local_blocks,
                                           local_block_size, local_stride);
      }
    }
    output[(query * num_heads + head) * kHeadDim + dim] =
        __float2bfloat16_rn(accum);
  }
}

__global__ void fp8_ds_mla_sparse_attention_grouped_kernel(
    const __mt_bfloat16* __restrict__ q, const uint8_t* __restrict__ cache,
    const void* __restrict__ indices, const void* __restrict__ lengths,
    const void* __restrict__ attn_sink, const uint8_t* __restrict__ extra_cache,
    const void* __restrict__ extra_indices,
    const void* __restrict__ extra_lengths, __mt_bfloat16* __restrict__ output,
    float* __restrict__ lse, int index_kind, int length_kind, int sink_kind,
    int64_t num_queries, int64_t seq_len, int64_t num_heads, int64_t q_dim,
    int64_t topk, int64_t extra_topk, int64_t num_blocks, int64_t block_size,
    int64_t block_stride, int64_t extra_num_blocks, int64_t extra_block_size,
    int64_t extra_block_stride, float softmax_scale) {
  const int64_t query = static_cast<int64_t>(blockIdx.x);
  const int64_t head_base =
      static_cast<int64_t>(blockIdx.y) * static_cast<int64_t>(kGroupedHeads);
  if (query >= num_queries || head_base >= num_heads) {
    return;
  }
  const int group_heads =
      static_cast<int>((num_heads - head_base) < kGroupedHeads
                           ? (num_heads - head_base)
                           : kGroupedHeads);

  extern __shared__ float shared[];
  float* staged = shared;
  float* slot_valid = staged + kGroupedTileSlots * kHeadDim;
  float* accum = slot_valid + kGroupedTileSlots;
  float* logits = accum + kGroupedHeads * kHeadDim;
  float* head_max = logits + kGroupedHeads * kGroupedTileSlots;
  float* head_sum = head_max + kGroupedHeads;
  float* head_has_valid = head_sum + kGroupedHeads;
  float* reduce = head_has_valid + kGroupedHeads;

  const int64_t total_topk = topk + extra_topk;
  const int64_t num_kv_tokens = num_blocks * block_size;
  const int64_t extra_num_kv_tokens = extra_num_blocks * extra_block_size;

  for (int64_t i = threadIdx.x;
       i < static_cast<int64_t>(kGroupedHeads) * kHeadDim;
       i += blockDim.x) {
    accum[i] = 0.0f;
  }
  if (threadIdx.x < kGroupedHeads) {
    head_max[threadIdx.x] = -INFINITY;
    head_sum[threadIdx.x] = 0.0f;
    head_has_valid[threadIdx.x] = 0.0f;
  }
  __syncthreads();

  for (int64_t tile_start = 0; tile_start < total_topk;
       tile_start += kGroupedTileSlots) {
    const int tile_slots =
        static_cast<int>((total_topk - tile_start) < kGroupedTileSlots
                             ? (total_topk - tile_start)
                             : kGroupedTileSlots);
    if (threadIdx.x < kGroupedTileSlots) {
      slot_valid[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int64_t i = threadIdx.x;
         i < static_cast<int64_t>(tile_slots) * kHeadDim; i += blockDim.x) {
      const int64_t local_slot = i / kHeadDim;
      const int64_t dim = i - local_slot * kHeadDim;
      const int64_t slot = tile_start + local_slot;
      int64_t raw_idx = 0;
      const bool is_extra = slot >= topk;
      const bool valid =
          is_extra
              ? slot_info(slot, query, topk, extra_topk, extra_num_kv_tokens,
                          indices, index_kind, lengths, length_kind,
                          extra_indices, extra_lengths, raw_idx)
              : slot_info(slot, query, topk, extra_topk, num_kv_tokens, indices,
                          index_kind, lengths, length_kind, extra_indices,
                          extra_lengths, raw_idx);
      if (dim == 0) {
        slot_valid[local_slot] = valid ? 1.0f : 0.0f;
      }
      float value = 0.0f;
      if (valid) {
        const uint8_t* local_cache = is_extra ? extra_cache : cache;
        const int64_t local_blocks = is_extra ? extra_num_blocks : num_blocks;
        const int64_t local_block_size =
            is_extra ? extra_block_size : block_size;
        const int64_t local_stride = is_extra ? extra_block_stride : block_stride;
        value = load_cache_value(local_cache, raw_idx, dim, local_blocks,
                                 local_block_size, local_stride);
      }
      staged[local_slot * kHeadDim + dim] = value;
    }
    __syncthreads();

    for (int local_head = 0; local_head < group_heads; ++local_head) {
      const int64_t head = head_base + local_head;
      float tile_max = -INFINITY;
      int tile_has_valid = 0;
      for (int slot = 0; slot < tile_slots; ++slot) {
        float partial = 0.0f;
        if (slot_valid[slot] > 0.5f) {
          for (int64_t dim = threadIdx.x; dim < q_dim; dim += blockDim.x) {
            const int64_t q_offset = (query * num_heads + head) * q_dim + dim;
            partial += __bfloat162float(q[q_offset]) *
                       staged[slot * kHeadDim + dim];
          }
        }
        reduce[threadIdx.x] = partial;
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
          if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
          }
          __syncthreads();
        }
        if (threadIdx.x == 0) {
          const float logit =
              slot_valid[slot] > 0.5f ? reduce[0] * softmax_scale : -INFINITY;
          logits[local_head * kGroupedTileSlots + slot] = logit;
          if (slot_valid[slot] > 0.5f) {
            tile_has_valid = 1;
            tile_max = fmaxf(tile_max, logit);
          }
        }
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        const bool old_has_valid = head_has_valid[local_head] > 0.5f;
        const bool new_has_valid = old_has_valid || tile_has_valid;
        const float old_max = head_max[local_head];
        const float old_sum = head_sum[local_head];
        const float new_max =
            new_has_valid ? fmaxf(old_max, tile_max) : -INFINITY;
        const float old_scale =
            old_has_valid ? expf(old_max - new_max) : 0.0f;
        float tile_sum = 0.0f;
        if (tile_has_valid) {
          for (int slot = 0; slot < tile_slots; ++slot) {
            if (slot_valid[slot] > 0.5f) {
              tile_sum +=
                  expf(logits[local_head * kGroupedTileSlots + slot] - new_max);
            }
          }
        }
        head_max[local_head] = new_max;
        head_sum[local_head] = old_sum * old_scale + tile_sum;
        head_has_valid[local_head] = new_has_valid ? 1.0f : 0.0f;
        reduce[0] = old_scale;
        reduce[1] = new_max;
      }
      __syncthreads();

      const float old_scale = reduce[0];
      const float current_max = reduce[1];
      for (int64_t dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
        accum[local_head * kHeadDim + dim] *= old_scale;
      }
      __syncthreads();

      for (int slot = 0; slot < tile_slots; ++slot) {
        if (slot_valid[slot] <= 0.5f) {
          continue;
        }
        const float weight =
            expf(logits[local_head * kGroupedTileSlots + slot] - current_max);
        for (int64_t dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
          accum[local_head * kHeadDim + dim] +=
              weight * staged[slot * kHeadDim + dim];
        }
        __syncthreads();
      }
    }
    __syncthreads();
  }

  for (int local_head = 0; local_head < group_heads; ++local_head) {
    const int64_t head = head_base + local_head;
    if (threadIdx.x == 0) {
      const bool has_valid = head_has_valid[local_head] > 0.5f;
      const float key_lse =
          has_valid ? head_max[local_head] + logf(head_sum[local_head])
                    : INFINITY;
      float denom = has_valid ? head_sum[local_head] : 0.0f;
      if (has_valid && attn_sink != nullptr) {
        denom += expf(load_sink(attn_sink, sink_kind, head) -
                      head_max[local_head]);
      }
      reduce[0] = has_valid && denom > 0.0f ? 1.0f / denom : 0.0f;
      const int64_t batch = query / seq_len;
      const int64_t seq = query - batch * seq_len;
      lse[(batch * num_heads + head) * seq_len + seq] = key_lse;
    }
    __syncthreads();

    const float norm = reduce[0];
    for (int64_t dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x) {
      const float value = accum[local_head * kHeadDim + dim] * norm;
      output[(query * num_heads + head) * kHeadDim + dim] =
          __float2bfloat16_rn(value);
    }
    __syncthreads();
  }
}

int index_kind(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == torch::kInt32) {
    return kIndexInt32;
  }
  if (tensor.scalar_type() == torch::kInt64) {
    return kIndexInt64;
  }
  TORCH_CHECK(false, "index tensors must be int32 or int64");
}

int optional_index_kind(const c10::optional<torch::Tensor>& tensor,
                        int fallback) {
  return tensor.has_value() ? index_kind(tensor.value()) : fallback;
}

const void* optional_data_ptr(const c10::optional<torch::Tensor>& tensor) {
  return tensor.has_value() ? tensor->data_ptr() : nullptr;
}

const uint8_t* optional_uint8_ptr(const c10::optional<torch::Tensor>& tensor) {
  return tensor.has_value() ? static_cast<const uint8_t*>(tensor->data_ptr())
                            : nullptr;
}

int sink_kind(const c10::optional<torch::Tensor>& tensor) {
  if (!tensor.has_value()) {
    return 0;
  }
  if (tensor->scalar_type() == torch::kFloat32) {
    return kSinkFloat32;
  }
  if (tensor->scalar_type() == torch::kBFloat16) {
    return kSinkBFloat16;
  }
  TORCH_CHECK(false, "attn_sink must be float32 or bfloat16");
}

void check_optional_index(const char* name,
                          const c10::optional<torch::Tensor>& tensor,
                          const torch::Tensor& device_ref) {
  if (!tensor.has_value()) {
    return;
  }
  TORCH_CHECK(tensor->device() == device_ref.device(),
              name, " must be on the same device as q");
  TORCH_CHECK(tensor->is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor->scalar_type() == torch::kInt32 ||
                  tensor->scalar_type() == torch::kInt64,
              name, " must be int32 or int64");
}

int64_t optional_topk(const c10::optional<torch::Tensor>& tensor) {
  return tensor.has_value() ? tensor->size(1) : 0;
}

}  // namespace

void fp8_ds_mla_sparse_attention(
    const torch::Tensor& q, const torch::Tensor& cache,
    const torch::Tensor& indices, const c10::optional<torch::Tensor>& lengths,
    const c10::optional<torch::Tensor>& attn_sink,
    const c10::optional<torch::Tensor>& extra_cache,
    const c10::optional<torch::Tensor>& extra_indices,
    const c10::optional<torch::Tensor>& extra_lengths, torch::Tensor& output,
    torch::Tensor& lse, double softmax_scale) {
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "output must be bfloat16");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");
  TORCH_CHECK(cache.scalar_type() == torch::kUInt8, "cache must be uint8");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 ||
                  indices.scalar_type() == torch::kInt64,
              "indices must be int32 or int64");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(cache.is_contiguous(), "cache must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
  TORCH_CHECK(q.device() == cache.device() && q.device() == indices.device() &&
                  q.device() == output.device() && q.device() == lse.device(),
              "q, cache, indices, output, and lse must be on the same device");
  TORCH_CHECK(q.dim() == 4, "q must be [batch, seq, heads, dim]");
  TORCH_CHECK(cache.dim() == 4 && cache.size(2) == 1,
              "cache must be [blocks, block, 1, bytes]");
  TORCH_CHECK(indices.dim() == 2, "indices must be [queries, topk]");
  TORCH_CHECK(output.dim() == 4 && output.size(0) == q.size(0) &&
                  output.size(1) == q.size(1) &&
                  output.size(2) == q.size(2) &&
                  output.size(3) == q.size(3),
              "output shape must match q");
  TORCH_CHECK(lse.dim() == 3 && lse.size(0) == q.size(0) &&
                  lse.size(1) == q.size(2) && lse.size(2) == q.size(1),
              "lse must be [batch, heads, seq]");
  TORCH_CHECK(q.size(3) == kHeadDim,
              "native sparse attention currently requires q dim 512");
  TORCH_CHECK(output.size(3) == kHeadDim,
              "native sparse attention currently requires output dim 512");
  TORCH_CHECK(indices.size(0) == q.size(0) * q.size(1),
              "indices must contain one row per query token");
  TORCH_CHECK(cache.size(0) > 0 && cache.size(1) > 0,
              "cache must contain at least one block");
  const int64_t block_size = cache.size(1);
  const int64_t expected_block_stride =
      block_size * (kTokenDataBytes + kTokenScaleBytes);
  TORCH_CHECK(cache.numel() % cache.size(0) == 0,
              "cache blocks must be evenly strided");
  TORCH_CHECK(cache.numel() / cache.size(0) >= expected_block_stride,
              "cache block payload is too small for fp8_ds_mla layout");

  check_optional_index("lengths", lengths, q);
  check_optional_index("extra_indices", extra_indices, q);
  check_optional_index("extra_lengths", extra_lengths, q);
  if (lengths.has_value()) {
    TORCH_CHECK(lengths->numel() == indices.size(0),
                "lengths must contain one value per query");
  }
  if (extra_indices.has_value()) {
    TORCH_CHECK(extra_cache.has_value(),
                "extra_cache is required when extra_indices is provided");
    TORCH_CHECK(extra_indices->dim() == 2 &&
                    extra_indices->size(0) == indices.size(0),
                "extra_indices must be [queries, extra_topk]");
  }
  if (extra_lengths.has_value()) {
    TORCH_CHECK(extra_indices.has_value(),
                "extra_indices is required when extra_lengths is provided");
    TORCH_CHECK(extra_lengths->numel() == indices.size(0),
                "extra_lengths must contain one value per query");
  }
  if (extra_cache.has_value()) {
    TORCH_CHECK(extra_indices.has_value(),
                "extra_indices is required when extra_cache is provided");
    TORCH_CHECK(extra_cache->device() == q.device(),
                "extra_cache must be on the same device as q");
    TORCH_CHECK(extra_cache->scalar_type() == torch::kUInt8,
                "extra_cache must be uint8");
    TORCH_CHECK(extra_cache->is_contiguous(), "extra_cache must be contiguous");
    TORCH_CHECK(extra_cache->dim() == 4 && extra_cache->size(2) == 1,
                "extra_cache must be [blocks, block, 1, bytes]");
    TORCH_CHECK(extra_cache->size(0) > 0 && extra_cache->size(1) > 0,
                "extra_cache must contain at least one block");
    TORCH_CHECK(extra_cache->numel() % extra_cache->size(0) == 0,
                "extra_cache blocks must be evenly strided");
    TORCH_CHECK(extra_cache->numel() / extra_cache->size(0) >=
                    extra_cache->size(1) *
                        (kTokenDataBytes + kTokenScaleBytes),
                "extra_cache block payload is too small for fp8_ds_mla layout");
  }
  if (attn_sink.has_value()) {
    TORCH_CHECK(attn_sink->device() == q.device(),
                "attn_sink must be on the same device as q");
    TORCH_CHECK(attn_sink->is_contiguous(), "attn_sink must be contiguous");
    TORCH_CHECK(attn_sink->numel() >= q.size(2),
                "attn_sink must contain at least one value per head");
    (void)sink_kind(attn_sink);
  }

  if (indices.numel() == 0) {
    return;
  }

  const int64_t extra_topk = optional_topk(extra_indices);
  const int64_t total_topk = indices.size(1) + extra_topk;
  TORCH_CHECK(total_topk > 0, "sparse attention requires at least one slot");
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>(indices.size(0)),
                  static_cast<unsigned int>(q.size(2)));
  const size_t shmem =
      static_cast<size_t>(total_topk + block.x) * sizeof(float);
  TORCH_CHECK(shmem <= 98304,
              "native sparse attention shared-memory request is too large");
  const int idx_kind = index_kind(indices);
  const int len_kind = optional_index_kind(lengths, idx_kind);
  if (extra_indices.has_value()) {
    TORCH_CHECK(index_kind(extra_indices.value()) == idx_kind,
                "extra_indices dtype must match indices dtype");
  }
  if (extra_lengths.has_value()) {
    TORCH_CHECK(index_kind(extra_lengths.value()) == len_kind,
                "extra_lengths dtype must match lengths dtype");
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  fp8_ds_mla_sparse_attention_kernel<<<grid, block, shmem, stream>>>(
      static_cast<const __mt_bfloat16*>(q.data_ptr()),
      static_cast<const uint8_t*>(cache.data_ptr()), indices.data_ptr(),
      optional_data_ptr(lengths), optional_data_ptr(attn_sink),
      optional_uint8_ptr(extra_cache), optional_data_ptr(extra_indices),
      optional_data_ptr(extra_lengths),
      static_cast<__mt_bfloat16*>(output.data_ptr()),
      static_cast<float*>(lse.data_ptr()), idx_kind, len_kind, sink_kind(attn_sink),
      indices.size(0), q.size(1), q.size(2), q.size(3), indices.size(1),
      extra_topk, cache.size(0), cache.size(1), cache.numel() / cache.size(0),
      extra_cache.has_value() ? extra_cache->size(0) : int64_t{0},
      extra_cache.has_value() ? extra_cache->size(1) : int64_t{1},
      extra_cache.has_value() ? extra_cache->numel() / extra_cache->size(0)
                              : int64_t{0},
      static_cast<float>(softmax_scale));
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "fp8_ds_mla_sparse_attention launch failed: ",
              musaGetErrorString(err));
}

void fp8_ds_mla_sparse_attention_grouped(
    const torch::Tensor& q, const torch::Tensor& cache,
    const torch::Tensor& indices, const c10::optional<torch::Tensor>& lengths,
    const c10::optional<torch::Tensor>& attn_sink,
    const c10::optional<torch::Tensor>& extra_cache,
    const c10::optional<torch::Tensor>& extra_indices,
    const c10::optional<torch::Tensor>& extra_lengths, torch::Tensor& output,
    torch::Tensor& lse, double softmax_scale) {
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "output must be bfloat16");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");
  TORCH_CHECK(cache.scalar_type() == torch::kUInt8, "cache must be uint8");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 ||
                  indices.scalar_type() == torch::kInt64,
              "indices must be int32 or int64");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(cache.is_contiguous(), "cache must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
  TORCH_CHECK(q.device() == cache.device() && q.device() == indices.device() &&
                  q.device() == output.device() && q.device() == lse.device(),
              "q, cache, indices, output, and lse must be on the same device");
  TORCH_CHECK(q.dim() == 4, "q must be [batch, seq, heads, dim]");
  TORCH_CHECK(cache.dim() == 4 && cache.size(2) == 1,
              "cache must be [blocks, block, 1, bytes]");
  TORCH_CHECK(indices.dim() == 2, "indices must be [queries, topk]");
  TORCH_CHECK(output.dim() == 4 && output.size(0) == q.size(0) &&
                  output.size(1) == q.size(1) &&
                  output.size(2) == q.size(2) &&
                  output.size(3) == q.size(3),
              "output shape must match q");
  TORCH_CHECK(lse.dim() == 3 && lse.size(0) == q.size(0) &&
                  lse.size(1) == q.size(2) && lse.size(2) == q.size(1),
              "lse must be [batch, heads, seq]");
  TORCH_CHECK(q.size(3) == kHeadDim,
              "grouped sparse attention currently requires q dim 512");
  TORCH_CHECK(output.size(3) == kHeadDim,
              "grouped sparse attention currently requires output dim 512");
  TORCH_CHECK(indices.size(0) == q.size(0) * q.size(1),
              "indices must contain one row per query token");
  TORCH_CHECK(cache.size(0) > 0 && cache.size(1) > 0,
              "cache must contain at least one block");
  const int64_t block_size = cache.size(1);
  const int64_t expected_block_stride =
      block_size * (kTokenDataBytes + kTokenScaleBytes);
  TORCH_CHECK(cache.numel() % cache.size(0) == 0,
              "cache blocks must be evenly strided");
  TORCH_CHECK(cache.numel() / cache.size(0) >= expected_block_stride,
              "cache block payload is too small for fp8_ds_mla layout");

  check_optional_index("lengths", lengths, q);
  check_optional_index("extra_indices", extra_indices, q);
  check_optional_index("extra_lengths", extra_lengths, q);
  if (lengths.has_value()) {
    TORCH_CHECK(lengths->numel() == indices.size(0),
                "lengths must contain one value per query");
  }
  if (extra_indices.has_value()) {
    TORCH_CHECK(extra_cache.has_value(),
                "extra_cache is required when extra_indices is provided");
    TORCH_CHECK(extra_indices->dim() == 2 &&
                    extra_indices->size(0) == indices.size(0),
                "extra_indices must be [queries, extra_topk]");
  }
  if (extra_lengths.has_value()) {
    TORCH_CHECK(extra_indices.has_value(),
                "extra_indices is required when extra_lengths is provided");
    TORCH_CHECK(extra_lengths->numel() == indices.size(0),
                "extra_lengths must contain one value per query");
  }
  if (extra_cache.has_value()) {
    TORCH_CHECK(extra_indices.has_value(),
                "extra_indices is required when extra_cache is provided");
    TORCH_CHECK(extra_cache->device() == q.device(),
                "extra_cache must be on the same device as q");
    TORCH_CHECK(extra_cache->scalar_type() == torch::kUInt8,
                "extra_cache must be uint8");
    TORCH_CHECK(extra_cache->is_contiguous(), "extra_cache must be contiguous");
    TORCH_CHECK(extra_cache->dim() == 4 && extra_cache->size(2) == 1,
                "extra_cache must be [blocks, block, 1, bytes]");
    TORCH_CHECK(extra_cache->size(0) > 0 && extra_cache->size(1) > 0,
                "extra_cache must contain at least one block");
    TORCH_CHECK(extra_cache->numel() % extra_cache->size(0) == 0,
                "extra_cache blocks must be evenly strided");
    TORCH_CHECK(extra_cache->numel() / extra_cache->size(0) >=
                    extra_cache->size(1) *
                        (kTokenDataBytes + kTokenScaleBytes),
                "extra_cache block payload is too small for fp8_ds_mla layout");
  }
  if (attn_sink.has_value()) {
    TORCH_CHECK(attn_sink->device() == q.device(),
                "attn_sink must be on the same device as q");
    TORCH_CHECK(attn_sink->is_contiguous(), "attn_sink must be contiguous");
    TORCH_CHECK(attn_sink->numel() >= q.size(2),
                "attn_sink must contain at least one value per head");
    (void)sink_kind(attn_sink);
  }

  if (indices.numel() == 0) {
    return;
  }

  const int64_t extra_topk = optional_topk(extra_indices);
  const int64_t total_topk = indices.size(1) + extra_topk;
  TORCH_CHECK(total_topk > 0, "sparse attention requires at least one slot");
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>(indices.size(0)),
                  static_cast<unsigned int>(
                      (q.size(2) + kGroupedHeads - 1) / kGroupedHeads));
  const size_t shmem =
      static_cast<size_t>(kGroupedTileSlots * kHeadDim + kGroupedTileSlots +
                          kGroupedHeads * kHeadDim +
                          kGroupedHeads * kGroupedTileSlots +
                          kGroupedHeads * 3 + block.x) *
      sizeof(float);
  TORCH_CHECK(shmem <= 98304,
              "grouped sparse attention shared-memory request is too large");
  const int idx_kind = index_kind(indices);
  const int len_kind = optional_index_kind(lengths, idx_kind);
  if (extra_indices.has_value()) {
    TORCH_CHECK(index_kind(extra_indices.value()) == idx_kind,
                "extra_indices dtype must match indices dtype");
  }
  if (extra_lengths.has_value()) {
    TORCH_CHECK(index_kind(extra_lengths.value()) == len_kind,
                "extra_lengths dtype must match lengths dtype");
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  fp8_ds_mla_sparse_attention_grouped_kernel<<<grid, block, shmem, stream>>>(
      static_cast<const __mt_bfloat16*>(q.data_ptr()),
      static_cast<const uint8_t*>(cache.data_ptr()), indices.data_ptr(),
      optional_data_ptr(lengths), optional_data_ptr(attn_sink),
      optional_uint8_ptr(extra_cache), optional_data_ptr(extra_indices),
      optional_data_ptr(extra_lengths),
      static_cast<__mt_bfloat16*>(output.data_ptr()),
      static_cast<float*>(lse.data_ptr()), idx_kind, len_kind, sink_kind(attn_sink),
      indices.size(0), q.size(1), q.size(2), q.size(3), indices.size(1),
      extra_topk, cache.size(0), cache.size(1), cache.numel() / cache.size(0),
      extra_cache.has_value() ? extra_cache->size(0) : int64_t{0},
      extra_cache.has_value() ? extra_cache->size(1) : int64_t{1},
      extra_cache.has_value() ? extra_cache->numel() / extra_cache->size(0)
                              : int64_t{0},
      static_cast<float>(softmax_scale));
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "fp8_ds_mla_sparse_attention_grouped launch failed: ",
              musaGetErrorString(err));
}
