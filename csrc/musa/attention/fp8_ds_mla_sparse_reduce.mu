#include <cmath>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

template <typename T>
__device__ __forceinline__ float load_as_float(const T* ptr, int64_t idx) {
  return static_cast<float>(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_as_float<__mt_bfloat16>(
    const __mt_bfloat16* ptr, int64_t idx) {
  return __bfloat162float(ptr[idx]);
}

template <>
__device__ __forceinline__ float load_as_float<__half>(const __half* ptr,
                                                       int64_t idx) {
  return __half2float(ptr[idx]);
}

template <typename T>
__device__ __forceinline__ void store_from_float(T* ptr, int64_t idx,
                                                 float value) {
  ptr[idx] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_from_float<__mt_bfloat16>(
    __mt_bfloat16* ptr, int64_t idx, float value) {
  ptr[idx] = __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ void store_from_float<__half>(__half* ptr,
                                                        int64_t idx,
                                                        float value) {
  ptr[idx] = __float2half_rn(value);
}

__device__ __forceinline__ float block_sum(float value, float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return shared[0];
}

template <typename QType, typename OutType, typename SinkType>
__global__ void fp8_ds_mla_sparse_reduce_kernel(
    const QType* __restrict__ q, const float* __restrict__ gathered,
    const bool* __restrict__ valid, const SinkType* __restrict__ attn_sink,
    bool has_attn_sink, float softmax_scale, OutType* __restrict__ output,
    float* __restrict__ lse, int64_t num_queries, int64_t num_heads,
    int64_t topk, int64_t q_dim, int64_t gathered_dim, int64_t value_dim) {
  const int64_t pair = static_cast<int64_t>(blockIdx.x);
  if (pair >= num_queries * num_heads) {
    return;
  }
  const int64_t query = pair / num_heads;
  const int64_t head = pair - query * num_heads;

  extern __shared__ float shared[];
  float* reduce_shared = shared;
  float* logits = shared + blockDim.x;

  const int64_t q_base = (query * num_heads + head) * q_dim;
  for (int64_t k = 0; k < topk; ++k) {
    float partial = 0.0f;
    const bool is_valid = valid[query * topk + k];
    const int64_t kv_base = (query * topk + k) * gathered_dim;
    if (is_valid) {
      for (int64_t dim = threadIdx.x; dim < q_dim; dim += blockDim.x) {
        partial += load_as_float(q, q_base + dim) * gathered[kv_base + dim];
      }
    }
    const float dot = block_sum(partial, reduce_shared);
    if (threadIdx.x == 0) {
      logits[k] = is_valid ? dot * softmax_scale : -INFINITY;
    }
    __syncthreads();
  }

  float max_logit = -INFINITY;
  for (int64_t k = 0; k < topk; ++k) {
    max_logit = fmaxf(max_logit, logits[k]);
  }
  const bool has_key = max_logit != -INFINITY;

  float sum_exp = 0.0f;
  if (has_key) {
    for (int64_t k = 0; k < topk; ++k) {
      if (logits[k] != -INFINITY) {
        sum_exp += expf(logits[k] - max_logit);
      }
    }
  }
  float key_lse = has_key ? max_logit + logf(sum_exp) : INFINITY;
  float lse_for_output = key_lse;
  if (has_attn_sink) {
    const float sink = load_as_float(attn_sink, head);
    if (has_key) {
      const float denom_max = fmaxf(key_lse, sink);
      lse_for_output =
          denom_max + logf(expf(key_lse - denom_max) +
                           expf(sink - denom_max));
    } else {
      lse_for_output = sink;
    }
  }

  if (threadIdx.x == 0) {
    lse[pair] = key_lse;
  }

  const int64_t out_base = pair * value_dim;
  for (int64_t dim = threadIdx.x; dim < value_dim; dim += blockDim.x) {
    float acc = 0.0f;
    if (has_key) {
      for (int64_t k = 0; k < topk; ++k) {
        const float logit = logits[k];
        if (logit != -INFINITY) {
          const float weight = expf(logit - lse_for_output);
          acc += weight * gathered[(query * topk + k) * gathered_dim + dim];
        }
      }
    }
    store_from_float(output, out_base + dim, acc);
  }
}

template <typename QType, typename OutType, typename SinkType>
void launch_fp8_ds_mla_sparse_reduce(const torch::Tensor& q,
                                     const torch::Tensor& gathered,
                                     const torch::Tensor& valid,
                                     const torch::Tensor* attn_sink,
                                     double softmax_scale,
                                     torch::Tensor& output,
                                     torch::Tensor& lse) {
  const int64_t num_queries = q.size(0);
  const int64_t num_heads = q.size(1);
  const int64_t q_dim = q.size(2);
  const int64_t topk = gathered.size(1);
  const int64_t gathered_dim = gathered.size(2);
  const int64_t value_dim = output.size(2);
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>(num_queries * num_heads));
  const int shmem =
      static_cast<int>((block.x + topk) * static_cast<int64_t>(sizeof(float)));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const SinkType* sink_ptr = attn_sink == nullptr
                                 ? nullptr
                                 : static_cast<const SinkType*>(
                                       attn_sink->data_ptr());
  fp8_ds_mla_sparse_reduce_kernel<QType, OutType, SinkType>
      <<<grid, block, shmem, stream>>>(
          static_cast<const QType*>(q.data_ptr()),
          static_cast<const float*>(gathered.data_ptr()),
          static_cast<const bool*>(valid.data_ptr()), sink_ptr,
          attn_sink != nullptr, static_cast<float>(softmax_scale),
          static_cast<OutType*>(output.data_ptr()),
          static_cast<float*>(lse.data_ptr()), num_queries, num_heads, topk,
          q_dim, gathered_dim, value_dim);
}

template <typename QType, typename OutType>
void dispatch_sink_type(const torch::Tensor& q, const torch::Tensor& gathered,
                        const torch::Tensor& valid,
                        const c10::optional<torch::Tensor>& attn_sink,
                        double softmax_scale, torch::Tensor& output,
                        torch::Tensor& lse) {
  if (!attn_sink.has_value()) {
    launch_fp8_ds_mla_sparse_reduce<QType, OutType, float>(
        q, gathered, valid, nullptr, softmax_scale, output, lse);
    return;
  }
  TORCH_CHECK(attn_sink->is_contiguous(), "attn_sink must be contiguous");
  TORCH_CHECK(attn_sink->numel() >= q.size(1),
              "attn_sink must contain at least one value per head");
  if (attn_sink->scalar_type() == torch::kFloat32) {
    launch_fp8_ds_mla_sparse_reduce<QType, OutType, float>(
        q, gathered, valid, &attn_sink.value(), softmax_scale, output, lse);
  } else if (attn_sink->scalar_type() == torch::kBFloat16) {
    launch_fp8_ds_mla_sparse_reduce<QType, OutType, __mt_bfloat16>(
        q, gathered, valid, &attn_sink.value(), softmax_scale, output, lse);
  } else if (attn_sink->scalar_type() == torch::kFloat16) {
    launch_fp8_ds_mla_sparse_reduce<QType, OutType, __half>(
        q, gathered, valid, &attn_sink.value(), softmax_scale, output, lse);
  } else {
    TORCH_CHECK(false, "attn_sink must be float32, bfloat16, or float16");
  }
}

template <typename QType>
void dispatch_output_type(const torch::Tensor& q, const torch::Tensor& gathered,
                          const torch::Tensor& valid,
                          const c10::optional<torch::Tensor>& attn_sink,
                          double softmax_scale, torch::Tensor& output,
                          torch::Tensor& lse) {
  if (output.scalar_type() == torch::kFloat32) {
    dispatch_sink_type<QType, float>(q, gathered, valid, attn_sink,
                                     softmax_scale, output, lse);
  } else if (output.scalar_type() == torch::kBFloat16) {
    dispatch_sink_type<QType, __mt_bfloat16>(q, gathered, valid, attn_sink,
                                             softmax_scale, output, lse);
  } else if (output.scalar_type() == torch::kFloat16) {
    dispatch_sink_type<QType, __half>(q, gathered, valid, attn_sink,
                                      softmax_scale, output, lse);
  } else {
    TORCH_CHECK(false, "output must be float32, bfloat16, or float16");
  }
}

}  // namespace

void fp8_ds_mla_sparse_reduce(const torch::Tensor& q,
                              const torch::Tensor& gathered,
                              const torch::Tensor& valid,
                              const c10::optional<torch::Tensor>& attn_sink,
                              double softmax_scale,
                              torch::Tensor& output,
                              torch::Tensor& lse) {
  TORCH_CHECK(q.dim() == 3, "q must be [num_queries, num_heads, q_dim]");
  TORCH_CHECK(gathered.dim() == 3,
              "gathered must be [num_queries, topk, dim]");
  TORCH_CHECK(valid.dim() == 2, "valid must be [num_queries, topk]");
  TORCH_CHECK(output.dim() == 3,
              "output must be [num_queries, num_heads, value_dim]");
  TORCH_CHECK(lse.dim() == 2, "lse must be [num_queries, num_heads]");
  TORCH_CHECK(gathered.scalar_type() == torch::kFloat32,
              "gathered must be float32");
  TORCH_CHECK(valid.scalar_type() == torch::kBool, "valid must be bool");
  TORCH_CHECK(lse.scalar_type() == torch::kFloat32, "lse must be float32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(gathered.is_contiguous(), "gathered must be contiguous");
  TORCH_CHECK(valid.is_contiguous(), "valid must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(lse.is_contiguous(), "lse must be contiguous");
  TORCH_CHECK(q.size(0) == gathered.size(0) && q.size(0) == valid.size(0) &&
                  q.size(0) == output.size(0) && q.size(0) == lse.size(0),
              "num_queries must match");
  TORCH_CHECK(q.size(1) == output.size(1) && q.size(1) == lse.size(1),
              "num_heads must match");
  TORCH_CHECK(gathered.size(1) == valid.size(1), "topk must match");
  TORCH_CHECK(gathered.size(2) >= q.size(2),
              "gathered dim must be at least q_dim");
  TORCH_CHECK(gathered.size(2) >= output.size(2),
              "gathered dim must be at least output value_dim");
  TORCH_CHECK(gathered.size(1) > 0, "topk must be positive");
  TORCH_CHECK(gathered.size(1) <= 8192,
              "topk is too large for the native sparse reduce scratch buffer");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 ||
                  q.scalar_type() == torch::kBFloat16 ||
                  q.scalar_type() == torch::kFloat16,
              "q must be float32, bfloat16, or float16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32 ||
                  output.scalar_type() == torch::kBFloat16 ||
                  output.scalar_type() == torch::kFloat16,
              "output must be float32, bfloat16, or float16");
  if (q.numel() == 0 || output.numel() == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(q));
  if (q.scalar_type() == torch::kFloat32) {
    dispatch_output_type<float>(q, gathered, valid, attn_sink, softmax_scale,
                                output, lse);
  } else if (q.scalar_type() == torch::kBFloat16) {
    dispatch_output_type<__mt_bfloat16>(q, gathered, valid, attn_sink,
                                        softmax_scale, output, lse);
  } else {
    dispatch_output_type<__half>(q, gathered, valid, attn_sink, softmax_scale,
                                 output, lse);
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "fp8_ds_mla_sparse_reduce launch failed: ",
              musaGetErrorString(err));
}
