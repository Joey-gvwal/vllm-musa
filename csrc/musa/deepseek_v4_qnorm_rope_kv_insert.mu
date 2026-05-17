#include <cmath>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp8.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int kHeadDim = 512;
constexpr int kNopeDim = 448;
constexpr int kRopeDim = 64;
constexpr int kHalfRopeDim = kRopeDim / 2;
constexpr int kQuantBlock = 64;
constexpr int kNumQuantBlocks = kNopeDim / kQuantBlock;
constexpr int kScaleBytesPerToken = kNumQuantBlocks + 1;
constexpr int kTokenValueBytes = kNopeDim + kRopeDim * 2;
constexpr float kFp8Max = 448.0f;
constexpr int kWarpSize = 32;
constexpr int kElemsPerLane = kHeadDim / kWarpSize;
constexpr uint32_t kFullMask = 0xffffffffu;

__device__ __forceinline__ float load_bf16(const __mt_bfloat16* ptr) {
  return __bfloat162float(*ptr);
}

__device__ __forceinline__ void store_bf16(__mt_bfloat16* ptr, float value) {
  *ptr = __float2bfloat16_rn(value);
}

__device__ __forceinline__ uint8_t fp8_e4m3_byte(float value) {
  __mt_fp8_e4m3 fp8_value(value);
  return static_cast<uint8_t>(fp8_value.__x);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value += __shfl_xor_sync(kFullMask, value, mask);
  }
  return value;
}

__device__ __forceinline__ float warp4_max_abs(float value) {
  float peer = __shfl_xor_sync(kFullMask, value, 1);
  value = fmaxf(value, peer);
  peer = __shfl_xor_sync(kFullMask, value, 2);
  return fmaxf(value, peer);
}

__global__ void fused_deepseek_v4_qnorm_rope_kv_insert_kernel(
    __mt_bfloat16* __restrict__ q,
    const __mt_bfloat16* __restrict__ kv,
    uint8_t* __restrict__ k_cache,
    const int64_t* __restrict__ slot_mapping,
    const int64_t* __restrict__ positions,
    const float* __restrict__ cos_sin_cache,
    float eps,
    int num_tokens_full,
    int num_tokens_insert,
    int num_heads,
    int block_size,
    int cache_stride_bytes) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int64_t global_slot =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  const int slots_per_token = num_heads + 1;
  const int token = static_cast<int>(global_slot / slots_per_token);
  const int head_slot = static_cast<int>(global_slot % slots_per_token);
  if (token >= num_tokens_full) {
    return;
  }

  const bool is_kv = head_slot == num_heads;
  if (is_kv && token >= num_tokens_insert) {
    return;
  }

  const int dim_base = lane * kElemsPerLane;
  float values[kElemsPerLane];

  if (head_slot < num_heads) {
    const int64_t q_offset =
        (static_cast<int64_t>(token) * num_heads + head_slot) * kHeadDim;
    const __mt_bfloat16* q_src = q + q_offset;
    __mt_bfloat16* q_dst = q + q_offset;

    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      values[i] = load_bf16(q_src + dim_base + i);
      sum_sq += values[i] * values[i];
    }

    const float inv_rms =
        rsqrtf(warp_sum(sum_sq) / static_cast<float>(kHeadDim) + eps);

    if (dim_base < kNopeDim) {
#pragma unroll
      for (int i = 0; i < kElemsPerLane; ++i) {
        store_bf16(q_dst + dim_base + i, values[i] * inv_rms);
      }
    } else {
      const int64_t pos = positions[token];
      const float* cos_sin = cos_sin_cache + pos * kRopeDim;
#pragma unroll
      for (int i = 0; i < kElemsPerLane; i += 2) {
        const int dim = dim_base + i;
        const int half_idx = (dim - kNopeDim) / 2;
        const float cos = cos_sin[half_idx];
        const float sin = cos_sin[kHalfRopeDim + half_idx];
        const float even = values[i] * inv_rms;
        const float odd = values[i + 1] * inv_rms;
        store_bf16(q_dst + dim, even * cos - odd * sin);
        store_bf16(q_dst + dim + 1, even * sin + odd * cos);
      }
    }
    return;
  }

  const int64_t slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }

  const int64_t block_idx = slot / block_size;
  const int64_t pos_in_block = slot % block_size;
  uint8_t* block_base =
      k_cache + block_idx * static_cast<int64_t>(cache_stride_bytes);
  uint8_t* token_fp8 = block_base + pos_in_block * kTokenValueBytes;
  uint8_t* token_bf16_bytes = token_fp8 + kNopeDim;
  uint8_t* token_scale =
      block_base + static_cast<int64_t>(block_size) * kTokenValueBytes +
      pos_in_block * kScaleBytesPerToken;
  const __mt_bfloat16* kv_src = kv + static_cast<int64_t>(token) * kHeadDim;

#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    values[i] = load_bf16(kv_src + dim_base + i);
  }

  float local_absmax = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    local_absmax = fmaxf(local_absmax, fabsf(values[i]));
  }

  const float amax = fmaxf(warp4_max_abs(local_absmax), 1.0e-4f);
  const float exponent = ceilf(log2f(amax / kFp8Max));
  const float inv_scale = exp2f(-exponent);

  if (dim_base < kNopeDim) {
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      const float scaled =
          fminf(fmaxf(values[i] * inv_scale, -kFp8Max), kFp8Max);
      token_fp8[dim_base + i] = fp8_e4m3_byte(scaled);
    }

    if ((lane & 3) == 0) {
      const int group = lane >> 2;
      const float encoded = fminf(fmaxf(exponent + 127.0f, 0.0f), 255.0f);
      token_scale[group] = static_cast<uint8_t>(encoded);
    }
    if (lane == 0) {
      token_scale[kNumQuantBlocks] = 0;
    }
  } else {
    const int64_t pos = positions[token];
    const float* cos_sin = cos_sin_cache + pos * kRopeDim;
    __mt_bfloat16* rope_dst =
        reinterpret_cast<__mt_bfloat16*>(token_bf16_bytes);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i += 2) {
      const int dim = dim_base + i;
      const int half_idx = (dim - kNopeDim) / 2;
      const float cos = cos_sin[half_idx];
      const float sin = cos_sin[kHalfRopeDim + half_idx];
      const float even = values[i];
      const float odd = values[i + 1];
      const int rope_dim = dim - kNopeDim;
      store_bf16(rope_dst + rope_dim, even * cos - odd * sin);
      store_bf16(rope_dst + rope_dim + 1, even * sin + odd * cos);
    }
  }
}

}  // namespace

void fused_deepseek_v4_qnorm_rope_kv_insert(
    torch::Tensor& q,
    const torch::Tensor& kv,
    torch::Tensor& k_cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    double eps,
    int64_t block_size) {
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kHeadDim,
              "q shape must be [N, H, 512]");
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == kHeadDim,
              "kv shape must be [N, 512]");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bfloat16");
  TORCH_CHECK(kv.scalar_type() == torch::kBFloat16, "kv must be bfloat16");
  TORCH_CHECK(k_cache.scalar_type() == torch::kUInt8,
              "k_cache must be uint8");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt64,
              "slot_mapping must be int64");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64,
              "positions must be int64");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32,
              "cos_sin_cache must be float32");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() &&
                  k_cache.is_contiguous(),
              "q, kv, and k_cache must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous() && positions.is_contiguous(),
              "slot_mapping and positions must be contiguous");
  TORCH_CHECK(kv.device() == q.device() && k_cache.device() == q.device() &&
                  slot_mapping.device() == q.device() &&
                  positions.device() == q.device() &&
                  cos_sin_cache.device() == q.device(),
              "kv, k_cache, slot_mapping, positions, and cos_sin_cache must "
              "be on the same device as q");
  TORCH_CHECK(cos_sin_cache.dim() == 2 &&
                  cos_sin_cache.size(1) == kRopeDim &&
                  cos_sin_cache.is_contiguous(),
              "cos_sin_cache must be contiguous with shape [max_pos, 64]");

  const int num_tokens = static_cast<int>(q.size(0));
  const int num_heads = static_cast<int>(q.size(1));
  TORCH_CHECK(kv.size(0) == num_tokens && positions.size(0) == num_tokens,
              "q, kv, and positions row counts must match");
  TORCH_CHECK(slot_mapping.size(0) <= num_tokens,
              "slot_mapping length cannot exceed q/kv rows");
  TORCH_CHECK(block_size > 0, "block_size must be positive");

  const int cache_stride = static_cast<int>(k_cache.stride(0));
  const at::musa::OptionalMUSAGuard device_guard(device_of(q));
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  const int block_threads = 256;
  const int warps_per_block = block_threads / kWarpSize;
  const int64_t total_slots =
      static_cast<int64_t>(num_tokens) * (num_heads + 1);
  const dim3 grid(
      static_cast<unsigned int>((total_slots + warps_per_block - 1) /
                                warps_per_block));
  const dim3 block(block_threads);
  fused_deepseek_v4_qnorm_rope_kv_insert_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<__mt_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __mt_bfloat16*>(kv.data_ptr()),
      reinterpret_cast<uint8_t*>(k_cache.data_ptr()),
      reinterpret_cast<const int64_t*>(slot_mapping.data_ptr()),
      reinterpret_cast<const int64_t*>(positions.data_ptr()),
      reinterpret_cast<const float*>(cos_sin_cache.data_ptr()),
      static_cast<float>(eps), num_tokens,
      static_cast<int>(slot_mapping.size(0)), num_heads,
      static_cast<int>(block_size), cache_stride);
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "fused_deepseek_v4_qnorm_rope_kv_insert launch failed: ",
              musaGetErrorString(err));
}
