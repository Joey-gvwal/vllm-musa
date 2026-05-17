#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

__device__ __forceinline__ float mxfp4_e2m1_value(uint8_t nibble) {
  const uint8_t sign = nibble & 0x08;
  const uint8_t mag = nibble & 0x07;
  float value = 0.0f;
  switch (mag) {
    case 0:
      value = 0.0f;
      break;
    case 1:
      value = 0.5f;
      break;
    case 2:
      value = 1.0f;
      break;
    case 3:
      value = 1.5f;
      break;
    case 4:
      value = 2.0f;
      break;
    case 5:
      value = 3.0f;
      break;
    case 6:
      value = 4.0f;
      break;
    default:
      value = 6.0f;
      break;
  }
  return sign ? -value : value;
}

__device__ __forceinline__ float e8m0_scale_to_float(uint8_t scale) {
  const uint32_t bits = static_cast<uint32_t>(scale) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

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

template <typename IdT, typename MapT>
__device__ __forceinline__ int64_t local_expert_id(const IdT* expert_ids,
                                                   const MapT* expert_map,
                                                   int64_t routed_row,
                                                   int64_t global_experts) {
  const int64_t global_expert = static_cast<int64_t>(expert_ids[routed_row]);
  if (global_expert < 0) {
    return -1;
  }
  if (expert_map == nullptr) {
    return global_expert;
  }
  if (global_expert >= global_experts) {
    return -1;
  }
  return static_cast<int64_t>(expert_map[global_expert]);
}

__device__ __forceinline__ float load_mxfp4(const uint8_t* packed,
                                            const uint8_t* scale,
                                            int64_t packed_base,
                                            int64_t scale_base,
                                            int64_t k) {
  const uint8_t byte = packed[packed_base + k / 2];
  const uint8_t nibble = (k & 1) == 0 ? (byte & 0x0f) : (byte >> 4);
  return mxfp4_e2m1_value(nibble) *
         e8m0_scale_to_float(scale[scale_base + k / 32]);
}

template <typename InT, typename WtT, typename IdT, typename MapT,
          typename OutT>
__global__ void mxfp4_naive_grouped_moe_kernel(
    const InT* __restrict__ hidden, const uint8_t* __restrict__ w1,
    const uint8_t* __restrict__ w2, const uint8_t* __restrict__ w1_scale,
    const uint8_t* __restrict__ w2_scale, const WtT* __restrict__ topk_weights,
    const IdT* __restrict__ topk_ids, const MapT* __restrict__ expert_map,
    OutT* __restrict__ output, int64_t num_tokens, int64_t hidden_dim,
    int64_t top_k, int64_t num_experts, int64_t global_experts,
    int64_t w1_out_dim, int64_t out_dim, bool apply_router_weight_on_input) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x);
  const int64_t token = static_cast<int64_t>(blockIdx.y);
  if (token >= num_tokens || out_col >= out_dim) {
    return;
  }

  const int64_t intermediate_dim = w1_out_dim / 2;
  const int64_t w1_packed_stride = hidden_dim / 2;
  const int64_t w1_scale_stride = hidden_dim / 32;
  const int64_t w2_packed_stride = intermediate_dim / 2;
  const int64_t w2_scale_stride = intermediate_dim / 32;

  float route_accum = 0.0f;

  for (int64_t route = 0; route < top_k; ++route) {
    const int64_t routed_row = token * top_k + route;
    const int64_t expert =
        local_expert_id(topk_ids, expert_map, routed_row, global_experts);
    if (expert < 0 || expert >= num_experts) {
      continue;
    }

    const float router_weight = load_as_float(topk_weights, routed_row);
    float partial = 0.0f;

    for (int64_t j = threadIdx.x; j < intermediate_dim; j += blockDim.x) {
      float gate = 0.0f;
      float up = 0.0f;
      const int64_t gate_base =
          (expert * w1_out_dim + j) * w1_packed_stride;
      const int64_t up_base =
          (expert * w1_out_dim + intermediate_dim + j) * w1_packed_stride;
      const int64_t gate_scale_base =
          (expert * w1_out_dim + j) * w1_scale_stride;
      const int64_t up_scale_base =
          (expert * w1_out_dim + intermediate_dim + j) * w1_scale_stride;

      for (int64_t k = 0; k < hidden_dim; ++k) {
        float x = load_as_float(hidden, token * hidden_dim + k);
        if (apply_router_weight_on_input) {
          x *= router_weight;
        }
        gate += x * load_mxfp4(w1, w1_scale, gate_base, gate_scale_base, k);
        up += x * load_mxfp4(w1, w1_scale, up_base, up_scale_base, k);
      }

      const float activated = silu(gate) * up;
      const int64_t w2_base =
          (expert * out_dim + out_col) * w2_packed_stride;
      const int64_t w2_scale_base =
          (expert * out_dim + out_col) * w2_scale_stride;
      partial +=
          activated * load_mxfp4(w2, w2_scale, w2_base, w2_scale_base, j);
    }

    extern __shared__ float shared[];
    shared[threadIdx.x] = partial;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        shared[threadIdx.x] += shared[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      route_accum += apply_router_weight_on_input ? shared[0]
                                                  : shared[0] * router_weight;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    store_from_float(output, token * out_dim + out_col, route_accum);
  }
}

template <typename InT, typename WtT, typename IdT, typename MapT,
          typename OutT>
void launch_mxfp4_naive_grouped_moe(
    const torch::Tensor& hidden, const torch::Tensor& w1,
    const torch::Tensor& w2, const torch::Tensor& w1_scale,
    const torch::Tensor& w2_scale, const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids, const torch::Tensor* expert_map,
    torch::Tensor& output, bool apply_router_weight_on_input) {
  const int64_t num_tokens = hidden.size(0);
  const int64_t hidden_dim = hidden.size(1);
  const int64_t top_k = topk_ids.size(1);
  const int64_t num_experts = w1.size(0);
  const int64_t w1_out_dim = w1.size(1);
  const int64_t out_dim = w2.size(1);
  const int64_t global_experts =
      expert_map == nullptr ? int64_t{0} : expert_map->numel();

  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>(out_dim),
                  static_cast<unsigned int>(num_tokens));
  const int shmem = static_cast<int>(block.x * sizeof(float));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const MapT* map_ptr = expert_map == nullptr
                            ? nullptr
                            : static_cast<const MapT*>(expert_map->data_ptr());
  mxfp4_naive_grouped_moe_kernel<InT, WtT, IdT, MapT, OutT>
      <<<grid, block, shmem, stream>>>(
          static_cast<const InT*>(hidden.data_ptr()),
          static_cast<const uint8_t*>(w1.data_ptr()),
          static_cast<const uint8_t*>(w2.data_ptr()),
          static_cast<const uint8_t*>(w1_scale.data_ptr()),
          static_cast<const uint8_t*>(w2_scale.data_ptr()),
          static_cast<const WtT*>(topk_weights.data_ptr()),
          static_cast<const IdT*>(topk_ids.data_ptr()), map_ptr,
          static_cast<OutT*>(output.data_ptr()), num_tokens, hidden_dim, top_k,
          num_experts, global_experts, w1_out_dim, out_dim,
          apply_router_weight_on_input);
}

template <typename InT, typename WtT, typename IdT, typename OutT>
void dispatch_map_type(const torch::Tensor& hidden, const torch::Tensor& w1,
                       const torch::Tensor& w2,
                       const torch::Tensor& w1_scale,
                       const torch::Tensor& w2_scale,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& topk_ids, torch::Tensor& output,
                       const c10::optional<torch::Tensor>& expert_map,
                       bool apply_router_weight_on_input) {
  if (!expert_map.has_value()) {
    launch_mxfp4_naive_grouped_moe<InT, WtT, IdT, int64_t, OutT>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, nullptr,
        output, apply_router_weight_on_input);
    return;
  }
  TORCH_CHECK(expert_map->is_contiguous(), "expert_map must be contiguous");
  TORCH_CHECK(expert_map->scalar_type() == torch::kInt32 ||
                  expert_map->scalar_type() == torch::kInt64,
              "expert_map must be int32 or int64");
  if (expert_map->scalar_type() == torch::kInt32) {
    launch_mxfp4_naive_grouped_moe<InT, WtT, IdT, int32_t, OutT>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
        &expert_map.value(), output, apply_router_weight_on_input);
  } else {
    launch_mxfp4_naive_grouped_moe<InT, WtT, IdT, int64_t, OutT>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids,
        &expert_map.value(), output, apply_router_weight_on_input);
  }
}

template <typename InT, typename WtT, typename OutT>
void dispatch_id_type(const torch::Tensor& hidden, const torch::Tensor& w1,
                      const torch::Tensor& w2,
                      const torch::Tensor& w1_scale,
                      const torch::Tensor& w2_scale,
                      const torch::Tensor& topk_weights,
                      const torch::Tensor& topk_ids, torch::Tensor& output,
                      const c10::optional<torch::Tensor>& expert_map,
                      bool apply_router_weight_on_input) {
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 ||
                  topk_ids.scalar_type() == torch::kInt64,
              "topk_ids must be int32 or int64");
  if (topk_ids.scalar_type() == torch::kInt32) {
    dispatch_map_type<InT, WtT, int32_t, OutT>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  } else {
    dispatch_map_type<InT, WtT, int64_t, OutT>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  }
}

template <typename InT, typename OutT>
void dispatch_weight_type(const torch::Tensor& hidden, const torch::Tensor& w1,
                          const torch::Tensor& w2,
                          const torch::Tensor& w1_scale,
                          const torch::Tensor& w2_scale,
                          const torch::Tensor& topk_weights,
                          const torch::Tensor& topk_ids,
                          torch::Tensor& output,
                          const c10::optional<torch::Tensor>& expert_map,
                          bool apply_router_weight_on_input) {
  if (topk_weights.scalar_type() == torch::kFloat32) {
    dispatch_id_type<InT, float, OutT>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  } else if (topk_weights.scalar_type() == torch::kBFloat16) {
    dispatch_id_type<InT, __mt_bfloat16, OutT>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  } else {
    TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat16,
                "topk_weights dtype must be float32, float16, or bfloat16");
    dispatch_id_type<InT, __half, OutT>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  }
}

template <typename InT>
void dispatch_output_type(const torch::Tensor& hidden, const torch::Tensor& w1,
                          const torch::Tensor& w2,
                          const torch::Tensor& w1_scale,
                          const torch::Tensor& w2_scale,
                          const torch::Tensor& topk_weights,
                          const torch::Tensor& topk_ids,
                          torch::Tensor& output,
                          const c10::optional<torch::Tensor>& expert_map,
                          bool apply_router_weight_on_input) {
  if (output.scalar_type() == torch::kFloat32) {
    dispatch_weight_type<InT, float>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  } else if (output.scalar_type() == torch::kBFloat16) {
    dispatch_weight_type<InT, __mt_bfloat16>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  } else {
    TORCH_CHECK(output.scalar_type() == torch::kFloat16,
                "output dtype must be float32, float16, or bfloat16");
    dispatch_weight_type<InT, __half>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  }
}

}  // namespace

void mxfp4_naive_grouped_moe(
    const torch::Tensor& hidden, const torch::Tensor& w1,
    const torch::Tensor& w2, const torch::Tensor& w1_scale,
    const torch::Tensor& w2_scale, const torch::Tensor& topk_weights,
    const torch::Tensor& topk_ids, torch::Tensor& output,
    const c10::optional<torch::Tensor>& expert_map,
    bool apply_router_weight_on_input) {
  TORCH_CHECK(hidden.dim() == 2, "hidden must be [num_tokens, hidden_dim]");
  TORCH_CHECK(w1.dim() == 3, "w1 must be [experts, 2*intermediate, hidden/2]");
  TORCH_CHECK(w2.dim() == 3, "w2 must be [experts, output, intermediate/2]");
  TORCH_CHECK(w1_scale.dim() == 3,
              "w1_scale must be [experts, 2*intermediate, hidden/32]");
  TORCH_CHECK(w2_scale.dim() == 3,
              "w2_scale must be [experts, output, intermediate/32]");
  TORCH_CHECK(topk_weights.dim() == 2, "topk_weights must be [tokens, top_k]");
  TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be [tokens, top_k]");
  TORCH_CHECK(output.dim() == 2, "output must be [tokens, output_dim]");
  TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
  TORCH_CHECK(w1.is_contiguous(), "w1 must be contiguous");
  TORCH_CHECK(w2.is_contiguous(), "w2 must be contiguous");
  TORCH_CHECK(w1_scale.is_contiguous(), "w1_scale must be contiguous");
  TORCH_CHECK(w2_scale.is_contiguous(), "w2_scale must be contiguous");
  TORCH_CHECK(topk_weights.is_contiguous(),
              "topk_weights must be contiguous");
  TORCH_CHECK(topk_ids.is_contiguous(), "topk_ids must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(w1.scalar_type() == torch::kUInt8, "w1 must be uint8");
  TORCH_CHECK(w2.scalar_type() == torch::kUInt8, "w2 must be uint8");
  TORCH_CHECK(w1_scale.scalar_type() == torch::kUInt8,
              "w1_scale must be uint8");
  TORCH_CHECK(w2_scale.scalar_type() == torch::kUInt8,
              "w2_scale must be uint8");
  TORCH_CHECK(hidden.size(0) == topk_weights.size(0) &&
                  hidden.size(0) == topk_ids.size(0) &&
                  hidden.size(0) == output.size(0),
              "hidden/topk/output token dimensions must match");
  TORCH_CHECK(topk_weights.size(1) == topk_ids.size(1),
              "topk_weights and topk_ids top-k dimensions must match");
  TORCH_CHECK(w1.size(0) == w2.size(0), "w1 and w2 expert counts must match");
  TORCH_CHECK(w1.size(1) % 2 == 0, "w1 output dimension must be even");
  TORCH_CHECK(w1.size(2) * 2 == hidden.size(1),
              "w1 packed input dimension must match hidden");
  TORCH_CHECK(w2.size(2) * 2 == w1.size(1) / 2,
              "w2 packed input dimension must match activated intermediate");
  TORCH_CHECK(w2.size(1) == output.size(1),
              "w2 output dimension must match output");
  TORCH_CHECK(w1_scale.size(0) == w1.size(0) &&
                  w1_scale.size(1) == w1.size(1) &&
                  w1_scale.size(2) * 32 == hidden.size(1),
              "w1_scale shape is incompatible with w1/hidden");
  TORCH_CHECK(w2_scale.size(0) == w2.size(0) &&
                  w2_scale.size(1) == w2.size(1) &&
                  w2_scale.size(2) * 32 == w1.size(1) / 2,
              "w2_scale shape is incompatible with w2/intermediate");

  TORCH_CHECK(w1.device() == hidden.device() && w2.device() == hidden.device() &&
                  w1_scale.device() == hidden.device() &&
                  w2_scale.device() == hidden.device() &&
                  topk_weights.device() == hidden.device() &&
                  topk_ids.device() == hidden.device() &&
                  output.device() == hidden.device(),
              "all tensors must be on the same device");
  if (expert_map.has_value()) {
    TORCH_CHECK(expert_map->device() == hidden.device(),
                "expert_map must be on the same device");
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(hidden));
  if (hidden.scalar_type() == torch::kFloat32) {
    dispatch_output_type<float>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  } else if (hidden.scalar_type() == torch::kBFloat16) {
    dispatch_output_type<__mt_bfloat16>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  } else {
    TORCH_CHECK(hidden.scalar_type() == torch::kFloat16,
                "hidden dtype must be float32, float16, or bfloat16");
    dispatch_output_type<__half>(
        hidden, w1, w2, w1_scale, w2_scale, topk_weights, topk_ids, output,
        expert_map, apply_router_weight_on_input);
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "mxfp4_naive_grouped_moe launch failed: ",
              musaGetErrorString(err));
}
