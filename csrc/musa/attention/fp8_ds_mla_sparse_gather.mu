#include <cmath>
#include <cstdint>

#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int64_t kNopeDim = 448;
constexpr int64_t kRopeDim = 64;
constexpr int64_t kOutDim = kNopeDim + kRopeDim;
constexpr int64_t kTokenDataBytes = kNopeDim + kRopeDim * 2;
constexpr int64_t kTokenScaleBytes = 8;
constexpr int64_t kQuantBlockSize = 64;

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

template <typename T>
__device__ __forceinline__ int64_t load_int(const T* ptr, int64_t idx) {
  return static_cast<int64_t>(ptr[idx]);
}

template <typename IndexT, typename LengthT>
__global__ void fp8_ds_mla_sparse_gather_kernel(
    const uint8_t* __restrict__ cache, const IndexT* __restrict__ indices,
    const LengthT* __restrict__ lengths, float* __restrict__ output,
    bool* __restrict__ valid_out, int64_t num_queries, int64_t topk,
    int64_t num_blocks, int64_t block_size, int64_t block_stride) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= num_queries * topk) {
    return;
  }

  const int64_t query = row / topk;
  const int64_t topk_idx = row - query * topk;
  const int64_t raw_idx = load_int(indices, row);
  const int64_t length =
      lengths == nullptr ? topk : load_int(lengths, query);
  const int64_t num_kv_tokens = num_blocks * block_size;
  const bool valid =
      raw_idx >= 0 && raw_idx < num_kv_tokens && topk_idx < length;
  const int64_t safe_idx = valid ? raw_idx : 0;
  const int64_t block_idx = safe_idx / block_size;
  const int64_t pos_in_block = safe_idx - block_idx * block_size;
  const uint8_t* block_ptr = cache + block_idx * block_stride;
  const int64_t token_base = pos_in_block * kTokenDataBytes;
  const int64_t scale_base =
      block_size * kTokenDataBytes + pos_in_block * kTokenScaleBytes;

  if (threadIdx.x == 0) {
    valid_out[row] = valid;
  }

  for (int64_t dim = threadIdx.x; dim < kOutDim; dim += blockDim.x) {
    float value = 0.0f;
    if (dim < kNopeDim) {
      const uint8_t fp8 = block_ptr[token_base + dim];
      const uint8_t scale_byte =
          block_ptr[scale_base + dim / kQuantBlockSize];
      value = fp8_e4m3fn_to_float(fp8) *
              ldexpf(1.0f, static_cast<int>(scale_byte) - 127);
    } else {
      const int64_t rope_dim = dim - kNopeDim;
      const uint8_t* bf16_ptr =
          block_ptr + token_base + kNopeDim + rope_dim * 2;
      value = bf16_bytes_to_float(bf16_ptr);
    }
    output[row * kOutDim + dim] = value;
  }
}

template <typename IndexT, typename LengthT>
void launch_fp8_ds_mla_sparse_gather(const torch::Tensor& cache,
                                     const torch::Tensor& indices,
                                     const torch::Tensor* lengths,
                                     torch::Tensor& output,
                                     torch::Tensor& valid) {
  const int64_t num_queries = indices.size(0);
  const int64_t topk = indices.size(1);
  const int64_t num_blocks = cache.size(0);
  const int64_t block_size = cache.size(1);
  const int64_t block_stride = cache.numel() / num_blocks;
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>(num_queries * topk));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const LengthT* length_ptr =
      lengths == nullptr ? nullptr : static_cast<const LengthT*>(lengths->data_ptr());
  fp8_ds_mla_sparse_gather_kernel<IndexT, LengthT><<<grid, block, 0, stream>>>(
      static_cast<const uint8_t*>(cache.data_ptr()),
      static_cast<const IndexT*>(indices.data_ptr()), length_ptr,
      static_cast<float*>(output.data_ptr()), static_cast<bool*>(valid.data_ptr()),
      num_queries, topk, num_blocks, block_size, block_stride);
}

template <typename IndexT>
void dispatch_length_type(const torch::Tensor& cache,
                          const torch::Tensor& indices,
                          const c10::optional<torch::Tensor>& lengths,
                          torch::Tensor& output, torch::Tensor& valid) {
  if (!lengths.has_value()) {
    launch_fp8_ds_mla_sparse_gather<IndexT, int64_t>(
        cache, indices, nullptr, output, valid);
    return;
  }
  TORCH_CHECK(lengths->is_contiguous(), "lengths must be contiguous");
  TORCH_CHECK(lengths->scalar_type() == torch::kInt32 ||
                  lengths->scalar_type() == torch::kInt64,
              "lengths must be int32 or int64");
  if (lengths->scalar_type() == torch::kInt32) {
    launch_fp8_ds_mla_sparse_gather<IndexT, int32_t>(
        cache, indices, &lengths.value(), output, valid);
  } else {
    launch_fp8_ds_mla_sparse_gather<IndexT, int64_t>(
        cache, indices, &lengths.value(), output, valid);
  }
}

}  // namespace

void fp8_ds_mla_sparse_gather(const torch::Tensor& cache,
                              const torch::Tensor& indices,
                              const c10::optional<torch::Tensor>& lengths,
                              torch::Tensor& output,
                              torch::Tensor& valid) {
  TORCH_CHECK(cache.scalar_type() == torch::kUInt8, "cache must be uint8");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32 ||
                  indices.scalar_type() == torch::kInt64,
              "indices must be int32 or int64");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32,
              "output must be float32");
  TORCH_CHECK(valid.scalar_type() == torch::kBool, "valid must be bool");
  TORCH_CHECK(cache.is_contiguous(), "cache must be contiguous");
  TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(valid.is_contiguous(), "valid must be contiguous");
  TORCH_CHECK(cache.dim() >= 2, "cache must have at least two dimensions");
  TORCH_CHECK(indices.dim() == 2, "indices must be [num_queries, topk]");
  TORCH_CHECK(output.dim() == 3,
              "output must be [num_queries, topk, 512]");
  TORCH_CHECK(valid.dim() == 2, "valid must be [num_queries, topk]");
  TORCH_CHECK(output.size(0) == indices.size(0) &&
                  output.size(1) == indices.size(1) &&
                  output.size(2) == kOutDim,
              "output shape must match indices and dim 512");
  TORCH_CHECK(valid.size(0) == indices.size(0) &&
                  valid.size(1) == indices.size(1),
              "valid shape must match indices");
  TORCH_CHECK(cache.size(0) > 0, "cache must have at least one block");
  TORCH_CHECK(cache.size(1) > 0, "cache block size must be positive");
  const int64_t block_size = cache.size(1);
  const int64_t expected_block_stride =
      block_size * (kTokenDataBytes + kTokenScaleBytes);
  TORCH_CHECK(cache.numel() % cache.size(0) == 0,
              "cache blocks must be evenly strided");
  TORCH_CHECK(cache.numel() / cache.size(0) >= expected_block_stride,
              "cache block payload is too small for fp8_ds_mla layout");
  if (lengths.has_value()) {
    TORCH_CHECK(lengths->numel() == indices.size(0),
                "lengths must contain one value per query");
  }

  if (indices.numel() == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(cache));
  if (indices.scalar_type() == torch::kInt32) {
    dispatch_length_type<int32_t>(cache, indices, lengths, output, valid);
  } else {
    dispatch_length_type<int64_t>(cache, indices, lengths, output, valid);
  }
  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "fp8_ds_mla_sparse_gather launch failed: ",
              musaGetErrorString(err));
}
