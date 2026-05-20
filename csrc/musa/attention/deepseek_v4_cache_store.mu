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

constexpr int64_t kNopeDim = 448;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kHeadDim = kNopeDim + kRopeDim;
constexpr int64_t kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr int64_t kTokenScaleBytes = 8;
constexpr int64_t kQuantBlockSize = 64;

constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ int64_t load_index(const void* ptr, int kind,
                                              int64_t idx) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t*>(ptr)[idx]);
  }
  return static_cast<int64_t>(static_cast<const int64_t*>(ptr)[idx]);
}

__global__ void deepseek_v4_store_sparse_kv_kernel(
    const __mt_bfloat16* __restrict__ normed, uint8_t* __restrict__ cache,
    const void* __restrict__ slots, const bool* __restrict__ write_mask,
    int index_kind, int64_t num_tokens, int64_t num_blocks, int64_t block_size,
    int64_t block_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) {
    return;
  }

  const int64_t slot = load_index(slots, index_kind, token);
  if (!write_mask[token] || slot < 0 || slot >= num_blocks * block_size) {
    return;
  }

  __shared__ float abs_values[kQuantBlockSize];
  __shared__ int scale_exponents[kTokenScaleBytes];

  const int tid = threadIdx.x;
  const int64_t block_idx = slot / block_size;
  const int64_t pos_in_block = slot - block_idx * block_size;
  uint8_t* block_ptr = cache + block_idx * block_stride;
  uint8_t* token_ptr = block_ptr + pos_in_block * kTokenDataBytes;
  uint8_t* scale_ptr = block_ptr + block_size * kTokenDataBytes +
                       pos_in_block * kTokenScaleBytes;
  const __mt_bfloat16* input = normed + token * kHeadDim;

  for (int qblock = 0; qblock < kNopeDim / kQuantBlockSize; ++qblock) {
    const int64_t start = qblock * kQuantBlockSize;
    if (tid < kQuantBlockSize) {
      const float value = __bfloat162float(input[start + tid]);
      abs_values[tid] = fabsf(value);
    }
    __syncthreads();

    for (int stride = kQuantBlockSize / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        abs_values[tid] = fmaxf(abs_values[tid], abs_values[tid + stride]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float amax = fmaxf(abs_values[0], 1.0e-4f);
      const int exponent =
          static_cast<int>(ceilf(log2f(amax / 448.0f)));
      scale_exponents[qblock] = exponent;
      const int scale_byte = max(0, min(255, exponent + 127));
      scale_ptr[qblock] = static_cast<uint8_t>(scale_byte);
    }
    __syncthreads();

    if (tid < kQuantBlockSize) {
      const float value = __bfloat162float(input[start + tid]);
      const float scaled = fminf(fmaxf(value * exp2f(-scale_exponents[qblock]),
                                      -448.0f),
                                448.0f);
      const __mt_fp8_e4m3 packed(scaled);
      token_ptr[start + tid] = packed.__x;
    }
    __syncthreads();
  }

  if (tid == 0) {
    scale_ptr[kTokenScaleBytes - 1] = 0;
  }
  if (tid < kRopeDim * 2) {
    const uint8_t* rope_bytes =
        reinterpret_cast<const uint8_t*>(input + kNopeDim);
    token_ptr[kNopeDim + tid] = rope_bytes[tid];
  }
}

int index_kind(const torch::Tensor& tensor) {
  if (tensor.scalar_type() == torch::kInt32) {
    return kIndexInt32;
  }
  if (tensor.scalar_type() == torch::kInt64) {
    return kIndexInt64;
  }
  TORCH_CHECK(false, "slot_mapping must be int32 or int64");
}

}  // namespace

void deepseek_v4_store_sparse_kv(
    const torch::Tensor& normed,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& write_mask) {
  TORCH_CHECK(normed.scalar_type() == torch::kBFloat16,
              "normed must be bfloat16");
  TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8,
              "kv_cache must be uint8");
  TORCH_CHECK(write_mask.scalar_type() == torch::kBool,
              "write_mask must be bool");
  TORCH_CHECK(normed.is_contiguous(), "normed must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  TORCH_CHECK(write_mask.is_contiguous(), "write_mask must be contiguous");
  TORCH_CHECK(normed.device() == kv_cache.device() &&
                  normed.device() == slot_mapping.device() &&
                  normed.device() == write_mask.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(normed.dim() == 2 && normed.size(1) == kHeadDim,
              "normed must be [num_tokens, 512]");
  TORCH_CHECK(kv_cache.dim() >= 2, "kv_cache must include block dimension");
  TORCH_CHECK(kv_cache.size(0) > 0 && kv_cache.size(1) > 0,
              "kv_cache must have non-empty blocks");
  TORCH_CHECK(slot_mapping.numel() == normed.size(0),
              "slot_mapping must have one entry per normed row");
  TORCH_CHECK(write_mask.numel() == normed.size(0),
              "write_mask must have one entry per normed row");
  TORCH_CHECK(kv_cache.stride(-1) == 1,
              "kv_cache byte dimension must be contiguous");
  const int64_t logical_block_bytes =
      kv_cache.size(1) * (kTokenDataBytes + kTokenScaleBytes);
  TORCH_CHECK(kv_cache.stride(0) >= logical_block_bytes,
              "kv_cache block stride is too small");

  if (normed.size(0) == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(normed));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const dim3 grid(static_cast<unsigned int>(normed.size(0)));
  const dim3 block(128);
  deepseek_v4_store_sparse_kv_kernel<<<grid, block, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(normed.data_ptr()),
      static_cast<uint8_t*>(kv_cache.data_ptr()), slot_mapping.data_ptr(),
      static_cast<const bool*>(write_mask.data_ptr()), index_kind(slot_mapping),
      normed.size(0), kv_cache.size(0), kv_cache.size(1), kv_cache.stride(0));
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "deepseek_v4_store_sparse_kv launch failed: ",
              musaGetErrorString(err));
}
