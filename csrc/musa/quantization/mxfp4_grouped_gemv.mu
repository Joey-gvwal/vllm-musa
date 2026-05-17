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
                                                   int64_t row,
                                                   int64_t global_experts) {
  const int64_t global_expert = static_cast<int64_t>(expert_ids[row]);
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

template <typename InT, typename OutT, typename IdT, typename MapT>
__global__ void mxfp4_grouped_gemv_kernel(
    const InT* __restrict__ input, const uint8_t* __restrict__ packed_weight,
    const uint8_t* __restrict__ weight_scale, const IdT* __restrict__ expert_ids,
    OutT* __restrict__ output, const MapT* __restrict__ expert_map,
    int64_t num_routed, int64_t in_dim, int64_t out_dim, int64_t num_experts,
    int64_t global_experts) {
  const int64_t out_col = static_cast<int64_t>(blockIdx.x);
  const int64_t row = static_cast<int64_t>(blockIdx.y);
  if (row >= num_routed || out_col >= out_dim) {
    return;
  }

  const int64_t expert = local_expert_id(expert_ids, expert_map, row,
                                         global_experts);
  if (expert < 0 || expert >= num_experts) {
    if (threadIdx.x == 0) {
      store_from_float(output, row * out_dim + out_col, 0.0f);
    }
    return;
  }

  float accum = 0.0f;
  const int64_t packed_stride = in_dim / 2;
  const int64_t scale_stride = in_dim / 32;
  const int64_t weight_base = (expert * out_dim + out_col) * packed_stride;
  const int64_t scale_base = (expert * out_dim + out_col) * scale_stride;
  const int64_t input_base = row * in_dim;

  for (int64_t k = threadIdx.x; k < in_dim; k += blockDim.x) {
    const uint8_t packed = packed_weight[weight_base + k / 2];
    const uint8_t nibble = (k & 1) == 0 ? (packed & 0x0f) : (packed >> 4);
    const uint8_t scale = weight_scale[scale_base + k / 32];
    const float w = mxfp4_e2m1_value(nibble) * e8m0_scale_to_float(scale);
    accum += load_as_float(input, input_base + k) * w;
  }

  extern __shared__ float shared[];
  shared[threadIdx.x] = accum;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    store_from_float(output, row * out_dim + out_col, shared[0]);
  }
}

template <typename InT, typename OutT, typename IdT, typename MapT>
void launch_mxfp4_grouped_gemv(const torch::Tensor& input,
                               const torch::Tensor& packed_weight,
                               const torch::Tensor& weight_scale,
                               const torch::Tensor& expert_ids,
                               torch::Tensor& output,
                               const torch::Tensor* expert_map) {
  const int64_t num_routed = input.size(0);
  const int64_t in_dim = input.size(1);
  const int64_t num_experts = packed_weight.size(0);
  const int64_t out_dim = packed_weight.size(1);
  const int64_t global_experts =
      expert_map == nullptr ? int64_t{0} : expert_map->numel();

  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>(out_dim),
                  static_cast<unsigned int>(num_routed));
  const int shmem = static_cast<int>(block.x * sizeof(float));
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  const MapT* map_ptr = expert_map == nullptr
                            ? nullptr
                            : static_cast<const MapT*>(expert_map->data_ptr());
  mxfp4_grouped_gemv_kernel<InT, OutT, IdT, MapT>
      <<<grid, block, shmem, stream>>>(
          static_cast<const InT*>(input.data_ptr()),
          static_cast<const uint8_t*>(packed_weight.data_ptr()),
          static_cast<const uint8_t*>(weight_scale.data_ptr()),
          static_cast<const IdT*>(expert_ids.data_ptr()),
          static_cast<OutT*>(output.data_ptr()), map_ptr, num_routed, in_dim,
          out_dim, num_experts, global_experts);
}

template <typename InT, typename OutT, typename IdT>
void dispatch_map_type(const torch::Tensor& input,
                       const torch::Tensor& packed_weight,
                       const torch::Tensor& weight_scale,
                       const torch::Tensor& expert_ids, torch::Tensor& output,
                       const c10::optional<torch::Tensor>& expert_map) {
  if (!expert_map.has_value()) {
    launch_mxfp4_grouped_gemv<InT, OutT, IdT, int64_t>(
        input, packed_weight, weight_scale, expert_ids, output, nullptr);
    return;
  }
  TORCH_CHECK(expert_map->is_contiguous(), "expert_map must be contiguous");
  TORCH_CHECK(expert_map->scalar_type() == torch::kInt32 ||
                  expert_map->scalar_type() == torch::kInt64,
              "expert_map must be int32 or int64");
  if (expert_map->scalar_type() == torch::kInt32) {
    launch_mxfp4_grouped_gemv<InT, OutT, IdT, int32_t>(
        input, packed_weight, weight_scale, expert_ids, output,
        &expert_map.value());
  } else {
    launch_mxfp4_grouped_gemv<InT, OutT, IdT, int64_t>(
        input, packed_weight, weight_scale, expert_ids, output,
        &expert_map.value());
  }
}

template <typename InT, typename OutT>
void dispatch_id_type(const torch::Tensor& input,
                      const torch::Tensor& packed_weight,
                      const torch::Tensor& weight_scale,
                      const torch::Tensor& expert_ids, torch::Tensor& output,
                      const c10::optional<torch::Tensor>& expert_map) {
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt32 ||
                  expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must be int32 or int64");
  if (expert_ids.scalar_type() == torch::kInt32) {
    dispatch_map_type<InT, OutT, int32_t>(
        input, packed_weight, weight_scale, expert_ids, output, expert_map);
  } else {
    dispatch_map_type<InT, OutT, int64_t>(
        input, packed_weight, weight_scale, expert_ids, output, expert_map);
  }
}

template <typename InT>
void dispatch_output_type(const torch::Tensor& input,
                          const torch::Tensor& packed_weight,
                          const torch::Tensor& weight_scale,
                          const torch::Tensor& expert_ids,
                          torch::Tensor& output,
                          const c10::optional<torch::Tensor>& expert_map) {
  if (output.scalar_type() == torch::kFloat32) {
    dispatch_id_type<InT, float>(input, packed_weight, weight_scale,
                                 expert_ids, output, expert_map);
  } else if (output.scalar_type() == torch::kBFloat16) {
    dispatch_id_type<InT, __mt_bfloat16>(input, packed_weight, weight_scale,
                                         expert_ids, output, expert_map);
  } else {
    TORCH_CHECK(output.scalar_type() == torch::kFloat16,
                "output dtype must be float32, float16, or bfloat16");
    dispatch_id_type<InT, __half>(input, packed_weight, weight_scale,
                                  expert_ids, output, expert_map);
  }
}

}  // namespace

void mxfp4_grouped_gemv(const torch::Tensor& input,
                        const torch::Tensor& packed_weight,
                        const torch::Tensor& weight_scale,
                        const torch::Tensor& expert_ids, torch::Tensor& output,
                        const c10::optional<torch::Tensor>& expert_map) {
  TORCH_CHECK(input.dim() == 2, "input must be [num_routed, in_dim]");
  TORCH_CHECK(packed_weight.dim() == 3,
              "packed_weight must be [experts, out_dim, in_dim / 2]");
  TORCH_CHECK(weight_scale.dim() == 3,
              "weight_scale must be [experts, out_dim, in_dim / 32]");
  TORCH_CHECK(expert_ids.dim() == 1, "expert_ids must be [num_routed]");
  TORCH_CHECK(output.dim() == 2, "output must be [num_routed, out_dim]");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
              "packed_weight must be uint8");
  TORCH_CHECK(weight_scale.scalar_type() == torch::kUInt8,
              "weight_scale must be uint8");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
  TORCH_CHECK(weight_scale.is_contiguous(), "weight_scale must be contiguous");
  TORCH_CHECK(expert_ids.is_contiguous(), "expert_ids must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(input.size(0) == expert_ids.size(0),
              "input rows must match expert_ids");
  TORCH_CHECK(output.size(0) == input.size(0),
              "output rows must match input rows");
  TORCH_CHECK(output.size(1) == packed_weight.size(1),
              "output columns must match packed_weight out_dim");
  TORCH_CHECK(input.size(1) == packed_weight.size(2) * 2,
              "input dim must equal packed_weight last dim * 2");
  TORCH_CHECK(input.size(1) % 32 == 0,
              "input dim must be divisible by MXFP4 block size 32");
  TORCH_CHECK(weight_scale.size(0) == packed_weight.size(0) &&
                  weight_scale.size(1) == packed_weight.size(1) &&
                  weight_scale.size(2) == input.size(1) / 32,
              "weight_scale shape must match packed weight scale blocks");
  TORCH_CHECK(input.scalar_type() == torch::kFloat32 ||
                  input.scalar_type() == torch::kFloat16 ||
                  input.scalar_type() == torch::kBFloat16,
              "input dtype must be float32, float16, or bfloat16");

  if (input.size(0) == 0 || output.size(1) == 0) {
    return;
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(input));
  if (input.scalar_type() == torch::kFloat32) {
    dispatch_output_type<float>(input, packed_weight, weight_scale, expert_ids,
                                output, expert_map);
  } else if (input.scalar_type() == torch::kBFloat16) {
    dispatch_output_type<__mt_bfloat16>(input, packed_weight, weight_scale,
                                        expert_ids, output, expert_map);
  } else {
    dispatch_output_type<__half>(input, packed_weight, weight_scale, expert_ids,
                                 output, expert_map);
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "mxfp4_grouped_gemv launch failed: ",
              musaGetErrorString(err));
}
