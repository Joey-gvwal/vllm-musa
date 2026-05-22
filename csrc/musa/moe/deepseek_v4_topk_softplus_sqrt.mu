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

constexpr int kIndexInt32 = 1;
constexpr int kIndexInt64 = 2;

__device__ __forceinline__ float load_scalar(const float* ptr,
                                             int64_t offset) {
  return ptr[offset];
}

__device__ __forceinline__ float load_scalar(const __half* ptr,
                                             int64_t offset) {
  return __half2float(ptr[offset]);
}

__device__ __forceinline__ float load_scalar(const __mt_bfloat16* ptr,
                                             int64_t offset) {
  return __bfloat162float(ptr[offset]);
}

__device__ __forceinline__ float sqrt_softplus(float x) {
  const float softplus = x > 20.0f ? x : log1pf(expf(x));
  return sqrtf(softplus);
}

__device__ __forceinline__ int64_t load_index(const void* ptr, int kind,
                                              int64_t offset) {
  if (kind == kIndexInt32) {
    return static_cast<int64_t>(static_cast<const int32_t*>(ptr)[offset]);
  }
  return static_cast<const int64_t*>(ptr)[offset];
}

__device__ __forceinline__ void store_index(void* ptr, int kind,
                                            int64_t offset, int64_t value) {
  if (kind == kIndexInt32) {
    static_cast<int32_t*>(ptr)[offset] = static_cast<int32_t>(value);
  } else {
    static_cast<int64_t*>(ptr)[offset] = value;
  }
}

template <typename InputT>
__global__ void deepseek_v4_topk_softplus_sqrt_kernel(
    const InputT* __restrict__ gating_output, float* __restrict__ topk_weights,
    void* __restrict__ topk_indices, int32_t* __restrict__ token_expert_indices,
    const float* __restrict__ correction_bias, const void* __restrict__ input_ids,
    const void* __restrict__ hash_indices_table, int topk_index_kind,
    int input_index_kind, int hash_index_kind, int64_t num_tokens,
    int64_t num_experts, int64_t topk, bool renormalize,
    float routed_scaling_factor, int64_t gating_stride_m,
    int64_t gating_stride_n, int64_t weights_stride_m,
    int64_t weights_stride_k, int64_t indices_stride_m,
    int64_t indices_stride_k, int64_t token_indices_stride_m,
    int64_t token_indices_stride_k, int64_t hash_stride_m,
    int64_t hash_stride_k) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x;
  extern __shared__ float shared[];
  float* scores = shared;
  float* choice_scores = shared + num_experts;
  float* reduce_values = choice_scores + num_experts;
  int32_t* reduce_indices =
      reinterpret_cast<int32_t*>(reduce_values + blockDim.x);

  if (token >= num_tokens) {
    return;
  }

  for (int64_t expert = tid; expert < num_experts; expert += blockDim.x) {
    const int64_t offset = token * gating_stride_m + expert * gating_stride_n;
    const float score = sqrt_softplus(load_scalar(gating_output, offset));
    scores[expert] = score;
    choice_scores[expert] =
        correction_bias == nullptr ? score : score + correction_bias[expert];
  }
  __syncthreads();

  float selected_sum = 0.0f;

  if (hash_indices_table != nullptr) {
    if (tid == 0) {
      const int64_t token_id = load_index(input_ids, input_index_kind, token);
      for (int64_t k_idx = 0; k_idx < topk; ++k_idx) {
        const int64_t expert = load_index(hash_indices_table, hash_index_kind,
                                          token_id * hash_stride_m +
                                              k_idx * hash_stride_k);
        const float score =
            expert >= 0 && expert < num_experts ? scores[expert] : 0.0f;
        const int64_t out_idx =
            token * weights_stride_m + k_idx * weights_stride_k;
        topk_weights[out_idx] = score;
        store_index(topk_indices, topk_index_kind,
                    token * indices_stride_m + k_idx * indices_stride_k, expert);
        token_expert_indices[token * token_indices_stride_m +
                             k_idx * token_indices_stride_k] =
            static_cast<int32_t>(expert);
        selected_sum += score;
      }
    }
  } else {
    for (int64_t k_idx = 0; k_idx < topk; ++k_idx) {
      int32_t local_expert = -1;
      float local_choice = -1.0e20f;
      for (int64_t expert = tid; expert < num_experts; expert += blockDim.x) {
        const float candidate = choice_scores[expert];
        if (candidate > local_choice ||
            (candidate == local_choice &&
             (local_expert < 0 || expert < local_expert))) {
          local_choice = candidate;
          local_expert = static_cast<int32_t>(expert);
        }
      }
      reduce_values[tid] = local_choice;
      reduce_indices[tid] = local_expert;
      __syncthreads();

      for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
          const float other_choice = reduce_values[tid + stride];
          const int32_t other_expert = reduce_indices[tid + stride];
          const bool other_wins =
              other_choice > reduce_values[tid] ||
              (other_choice == reduce_values[tid] && other_expert >= 0 &&
               (reduce_indices[tid] < 0 || other_expert < reduce_indices[tid]));
          if (other_wins) {
            reduce_values[tid] = other_choice;
            reduce_indices[tid] = other_expert;
          }
        }
        __syncthreads();
      }

      const int32_t best_expert = reduce_indices[0];

      if (tid == 0) {
        const float score = scores[best_expert];
        const int64_t out_idx =
            token * weights_stride_m + k_idx * weights_stride_k;
        topk_weights[out_idx] = score;
        store_index(topk_indices, topk_index_kind,
                    token * indices_stride_m + k_idx * indices_stride_k,
                    best_expert);
        token_expert_indices[token * token_indices_stride_m +
                             k_idx * token_indices_stride_k] = best_expert;
        selected_sum += score;
        choice_scores[best_expert] = -1.0e20f;
      }
      __syncthreads();
    }
  }

  if (tid == 0) {
    float scale = routed_scaling_factor;
    if (renormalize) {
      const float denom = selected_sum > 0.0f ? selected_sum : 1.0f;
      scale /= denom;
    }
    if (scale != 1.0f) {
      for (int64_t k_idx = 0; k_idx < topk; ++k_idx) {
        const int64_t out_idx =
            token * weights_stride_m + k_idx * weights_stride_k;
        topk_weights[out_idx] *= scale;
      }
    }
  }
}

void check_musa_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name,
              " must be a MUSA tensor");
}

int index_kind(const torch::Tensor& tensor, const char* name) {
  if (tensor.scalar_type() == torch::kInt32) {
    return kIndexInt32;
  }
  if (tensor.scalar_type() == torch::kInt64) {
    return kIndexInt64;
  }
  TORCH_CHECK(false, name, " must be int32 or int64");
}

int choose_threads(int64_t num_experts) {
  if (num_experts <= 64) {
    return 64;
  }
  if (num_experts <= 128) {
    return 128;
  }
  if (num_experts <= 256) {
    return 256;
  }
  if (num_experts <= 512) {
    return 512;
  }
  return 1024;
}

template <typename InputT>
void launch_deepseek_v4_topk_softplus_sqrt(
    const torch::Tensor& gating_output, torch::Tensor& topk_weights,
    torch::Tensor& topk_indices, torch::Tensor& token_expert_indices,
    bool renormalize, double routed_scaling_factor,
    const c10::optional<torch::Tensor>& correction_bias,
    const c10::optional<torch::Tensor>& input_ids,
    const c10::optional<torch::Tensor>& hash_indices_table,
    musaStream_t stream) {
  const int64_t num_tokens = gating_output.size(0);
  const int64_t num_experts = gating_output.size(1);
  const int64_t topk = topk_weights.size(1);
  const int threads = choose_threads(num_experts);
  const size_t shared_bytes =
      static_cast<size_t>(num_experts) * 2 * sizeof(float) +
      static_cast<size_t>(threads) * (sizeof(float) + sizeof(int32_t));
  const float* bias_ptr =
      correction_bias.has_value() ? correction_bias.value().data_ptr<float>()
                                  : nullptr;
  const void* input_ids_ptr =
      input_ids.has_value() ? input_ids.value().data_ptr() : nullptr;
  const void* hash_table_ptr = hash_indices_table.has_value()
                                   ? hash_indices_table.value().data_ptr()
                                   : nullptr;
  const int input_index_kind =
      input_ids.has_value() ? index_kind(input_ids.value(), "input_ids")
                            : kIndexInt32;
  const int hash_index_kind = hash_indices_table.has_value()
                                  ? index_kind(hash_indices_table.value(),
                                               "hash_indices_table")
                                  : kIndexInt32;
  const int64_t hash_stride_m =
      hash_indices_table.has_value() ? hash_indices_table.value().stride(0) : 0;
  const int64_t hash_stride_k =
      hash_indices_table.has_value() ? hash_indices_table.value().stride(1) : 0;

  deepseek_v4_topk_softplus_sqrt_kernel<InputT>
      <<<static_cast<unsigned int>(num_tokens), threads, shared_bytes, stream>>>(
          reinterpret_cast<const InputT*>(gating_output.data_ptr()),
          topk_weights.data_ptr<float>(), topk_indices.data_ptr(),
          token_expert_indices.data_ptr<int32_t>(), bias_ptr, input_ids_ptr,
          hash_table_ptr, index_kind(topk_indices, "topk_indices"),
          input_index_kind, hash_index_kind, num_tokens, num_experts, topk,
          renormalize, static_cast<float>(routed_scaling_factor),
          gating_output.stride(0), gating_output.stride(1),
          topk_weights.stride(0), topk_weights.stride(1),
          topk_indices.stride(0), topk_indices.stride(1),
          token_expert_indices.stride(0), token_expert_indices.stride(1),
          hash_stride_m, hash_stride_k);
}

}  // namespace

void deepseek_v4_topk_softplus_sqrt(
    torch::Tensor& topk_weights, torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices, const torch::Tensor& gating_output,
    bool renormalize, double routed_scaling_factor,
    const c10::optional<torch::Tensor>& correction_bias,
    const c10::optional<torch::Tensor>& input_ids,
    const c10::optional<torch::Tensor>& hash_indices_table) {
  check_musa_tensor(topk_weights, "topk_weights");
  check_musa_tensor(topk_indices, "topk_indices");
  check_musa_tensor(token_expert_indices, "token_expert_indices");
  check_musa_tensor(gating_output, "gating_output");
  TORCH_CHECK(gating_output.dim() == 2,
              "gating_output must have shape [num_tokens, num_experts]");
  TORCH_CHECK(topk_weights.dim() == 2 && topk_indices.dim() == 2 &&
                  token_expert_indices.dim() == 2,
              "topk outputs must have shape [num_tokens, topk]");
  TORCH_CHECK(topk_weights.sizes() == topk_indices.sizes() &&
                  topk_weights.sizes() == token_expert_indices.sizes(),
              "topk output tensors must have matching shapes");
  TORCH_CHECK(topk_weights.size(0) == gating_output.size(0),
              "topk output row count must match gating_output");
  TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32,
              "topk_weights must be float32");
  TORCH_CHECK(token_expert_indices.scalar_type() == torch::kInt32,
              "token_expert_indices must be int32");
  index_kind(topk_indices, "topk_indices");
  TORCH_CHECK(gating_output.size(1) > 0 && gating_output.size(1) <= 1024,
              "deepseek_v4_topk_softplus_sqrt supports 1..1024 experts");
  TORCH_CHECK(topk_weights.size(1) > 0 && topk_weights.size(1) <= 64,
              "deepseek_v4_topk_softplus_sqrt supports topk in 1..64");
  TORCH_CHECK(topk_weights.device() == gating_output.device() &&
                  topk_indices.device() == gating_output.device() &&
                  token_expert_indices.device() == gating_output.device(),
              "all tensors must be on the same device");

  if (correction_bias.has_value()) {
    check_musa_tensor(correction_bias.value(), "correction_bias");
    TORCH_CHECK(correction_bias.value().scalar_type() == torch::kFloat32,
                "correction_bias must be float32");
    TORCH_CHECK(correction_bias.value().dim() == 1 &&
                    correction_bias.value().size(0) == gating_output.size(1),
                "correction_bias must have shape [num_experts]");
    TORCH_CHECK(correction_bias.value().is_contiguous(),
                "correction_bias must be contiguous");
  }

  if (hash_indices_table.has_value()) {
    check_musa_tensor(hash_indices_table.value(), "hash_indices_table");
    TORCH_CHECK(input_ids.has_value(),
                "input_ids is required when hash_indices_table is provided");
    check_musa_tensor(input_ids.value(), "input_ids");
    index_kind(hash_indices_table.value(), "hash_indices_table");
    index_kind(input_ids.value(), "input_ids");
    TORCH_CHECK(input_ids.value().dim() == 1 &&
                    input_ids.value().size(0) == gating_output.size(0),
                "input_ids must have shape [num_tokens]");
    TORCH_CHECK(hash_indices_table.value().dim() == 2 &&
                    hash_indices_table.value().size(1) >= topk_weights.size(1),
                "hash_indices_table must have shape [vocab_size, >=topk]");
  }

  if (gating_output.numel() == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(gating_output));
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  if (gating_output.scalar_type() == torch::kFloat32) {
    launch_deepseek_v4_topk_softplus_sqrt<float>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        renormalize, routed_scaling_factor, correction_bias, input_ids,
        hash_indices_table, stream);
  } else if (gating_output.scalar_type() == torch::kFloat16) {
    launch_deepseek_v4_topk_softplus_sqrt<__half>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        renormalize, routed_scaling_factor, correction_bias, input_ids,
        hash_indices_table, stream);
  } else if (gating_output.scalar_type() == torch::kBFloat16) {
    launch_deepseek_v4_topk_softplus_sqrt<__mt_bfloat16>(
        gating_output, topk_weights, topk_indices, token_expert_indices,
        renormalize, routed_scaling_factor, correction_bias, input_ids,
        hash_indices_table, stream);
  } else {
    TORCH_CHECK(false, "Unsupported gating_output data type: ",
                gating_output.scalar_type());
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_topk_softplus_sqrt launch failed: ",
              musaGetErrorString(err));
}
