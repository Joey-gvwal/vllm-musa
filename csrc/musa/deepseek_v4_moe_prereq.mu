#include <cmath>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_fp8.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr float kFp8E4m3Max = 448.0f;
constexpr int kUe8m0Bias = 127;

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

__device__ __forceinline__ uint8_t amax_to_ue8m0_scale(float amax) {
  const float raw_scale = fmaxf(amax, 1.0e-10f) / kFp8E4m3Max;
  int exp = static_cast<int>(ceilf(log2f(raw_scale))) + kUe8m0Bias;
  if (exp < 0) {
    exp = 0;
  }
  if (exp > 255) {
    exp = 255;
  }
  return static_cast<uint8_t>(exp);
}

__device__ __forceinline__ float ue8m0_to_float(uint8_t scale) {
  const uint32_t bits = static_cast<uint32_t>(scale) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ void write_scale_byte(void* scale_ptr,
                                                 int64_t byte_idx,
                                                 uint8_t value) {
  reinterpret_cast<uint8_t*>(scale_ptr)[byte_idx] = value;
}

__device__ __forceinline__ void store_fp8(void* output, int64_t idx,
                                          float value) {
  const float clamped = fminf(fmaxf(value, -kFp8E4m3Max), kFp8E4m3Max);
  reinterpret_cast<__mt_fp8_storage_t*>(output)[idx] =
      __musa_cvt_float_to_fp8(clamped, __MT_SATFINITE, __MT_E4M3);
}

template <typename InT>
__global__ void mega_moe_quant_kernel(const InT* __restrict__ x,
                                      void* __restrict__ buf_x,
                                      void* __restrict__ buf_x_sf,
                                      int64_t num_tokens, int64_t hidden,
                                      int64_t group_size,
                                      int64_t num_groups) {
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t row = block / num_groups;
  const int64_t group = block - row * num_groups;
  if (row >= num_tokens) {
    return;
  }

  extern __shared__ float shared[];
  float local_max = 0.0f;
  const int64_t base = row * hidden + group * group_size;
  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float value = load_as_float(x, base + i);
    local_max = fmaxf(local_max, fabsf(value));
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x],
                                  shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const uint8_t scale_byte = amax_to_ue8m0_scale(shared[0]);
  const float scale = ue8m0_to_float(scale_byte);
  if (threadIdx.x == 0) {
    write_scale_byte(buf_x_sf, row * num_groups + group, scale_byte);
  }

  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    const float value = load_as_float(x, base + i) / scale;
    store_fp8(buf_x, base + i, value);
  }
}

template <typename TopkInT, typename TopkOutT>
__global__ void mega_moe_topk_kernel(
    const TopkInT* __restrict__ topk_idx,
    const float* __restrict__ topk_weights,
    TopkOutT* __restrict__ buf_topk_idx,
    float* __restrict__ buf_topk_weights, int64_t num_tokens,
    int64_t padded_max, int64_t top_k) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t col = static_cast<int64_t>(threadIdx.x);
  if (row >= padded_max || col >= top_k) {
    return;
  }
  const int64_t idx = row * top_k + col;
  if (row < num_tokens) {
    buf_topk_idx[idx] = static_cast<TopkOutT>(topk_idx[idx]);
    buf_topk_weights[idx] = topk_weights[idx];
  } else {
    buf_topk_idx[idx] = static_cast<TopkOutT>(-1);
    buf_topk_weights[idx] = 0.0f;
  }
}

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

template <typename InT>
__global__ void swiglu_post_quant_2d_kernel(
    const InT* __restrict__ input, void* __restrict__ output,
    void* __restrict__ output_scale, const int64_t* __restrict__ masked_m,
    int64_t rows, int64_t hidden, int64_t group_size, int64_t num_groups,
    float swiglu_limit) {
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t row = block / num_groups;
  const int64_t group = block - row * num_groups;
  if (row >= rows || row >= masked_m[0]) {
    return;
  }

  extern __shared__ float shared[];
  float local_max = 0.0f;
  const int64_t out_base = row * hidden + group * group_size;
  const int64_t in_base = row * hidden * 2 + group * group_size;
  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    float gate = load_as_float(input, in_base + i);
    float up = load_as_float(input, in_base + hidden + i);
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    const float value = silu(gate) * up;
    shared[threadIdx.x] = fmaxf(local_max, fabsf(value));
    local_max = shared[threadIdx.x];
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x],
                                  shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const uint8_t scale_byte = amax_to_ue8m0_scale(shared[0]);
  const float scale = ue8m0_to_float(scale_byte);
  if (threadIdx.x == 0) {
    write_scale_byte(output_scale, row * num_groups + group, scale_byte);
  }

  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    float gate = load_as_float(input, in_base + i);
    float up = load_as_float(input, in_base + hidden + i);
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    store_fp8(output, out_base + i, (silu(gate) * up) / scale);
  }
}

template <typename InT>
__global__ void swiglu_post_quant_3d_kernel(
    const InT* __restrict__ input, void* __restrict__ output,
    void* __restrict__ output_scale, const int64_t* __restrict__ masked_m,
    int64_t experts, int64_t rows, int64_t hidden, int64_t group_size,
    int64_t num_groups, float swiglu_limit) {
  const int64_t block = static_cast<int64_t>(blockIdx.x);
  const int64_t expert = block / (rows * num_groups);
  const int64_t rem = block - expert * rows * num_groups;
  const int64_t row = rem / num_groups;
  const int64_t group = rem - row * num_groups;
  if (expert >= experts || row >= rows || row >= masked_m[expert]) {
    return;
  }

  extern __shared__ float shared[];
  float local_max = 0.0f;
  const int64_t out_base = (expert * rows + row) * hidden + group * group_size;
  const int64_t in_base =
      (expert * rows + row) * hidden * 2 + group * group_size;
  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    float gate = load_as_float(input, in_base + i);
    float up = load_as_float(input, in_base + hidden + i);
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    const float value = silu(gate) * up;
    local_max = fmaxf(local_max, fabsf(value));
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] = fmaxf(shared[threadIdx.x],
                                  shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const uint8_t scale_byte = amax_to_ue8m0_scale(shared[0]);
  const float scale = ue8m0_to_float(scale_byte);
  if (threadIdx.x == 0) {
    write_scale_byte(output_scale,
                     (expert * rows + row) * num_groups + group, scale_byte);
  }

  for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
    float gate = load_as_float(input, in_base + i);
    float up = load_as_float(input, in_base + hidden + i);
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    store_fp8(output, out_base + i, (silu(gate) * up) / scale);
  }
}

template <typename InT>
void launch_mega_moe_quant(const torch::Tensor& x, torch::Tensor& buf_x,
                           torch::Tensor& buf_x_sf, int64_t group_size) {
  const int64_t num_tokens = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t num_groups = hidden / group_size;
  if (num_tokens == 0) {
    return;
  }
  const dim3 block(128);
  const dim3 grid(static_cast<unsigned int>(num_tokens * num_groups));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  mega_moe_quant_kernel<InT><<<grid, block, block.x * sizeof(float), stream>>>(
      static_cast<const InT*>(x.data_ptr()), buf_x.data_ptr(),
      buf_x_sf.data_ptr(), num_tokens, hidden, group_size, num_groups);
}

template <typename TopkInT>
void launch_topk_copy(const torch::Tensor& topk_idx,
                      const torch::Tensor& topk_weights,
                      torch::Tensor& buf_topk_idx,
                      torch::Tensor& buf_topk_weights) {
  const int64_t num_tokens = topk_idx.size(0);
  const int64_t padded_max = buf_topk_idx.size(0);
  const int64_t top_k = topk_idx.size(1);
  if (padded_max == 0 || top_k == 0) {
    return;
  }
  const dim3 block(static_cast<unsigned int>(top_k));
  const dim3 grid(static_cast<unsigned int>(padded_max));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  if (buf_topk_idx.scalar_type() == torch::kInt32) {
    mega_moe_topk_kernel<TopkInT, int32_t><<<grid, block, 0, stream>>>(
        static_cast<const TopkInT*>(topk_idx.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()),
        static_cast<int32_t*>(buf_topk_idx.data_ptr()),
        static_cast<float*>(buf_topk_weights.data_ptr()), num_tokens,
        padded_max, top_k);
  } else {
    mega_moe_topk_kernel<TopkInT, int64_t><<<grid, block, 0, stream>>>(
        static_cast<const TopkInT*>(topk_idx.data_ptr()),
        static_cast<const float*>(topk_weights.data_ptr()),
        static_cast<int64_t*>(buf_topk_idx.data_ptr()),
        static_cast<float*>(buf_topk_weights.data_ptr()), num_tokens,
        padded_max, top_k);
  }
}

template <typename InT>
void launch_swiglu_post_quant(const torch::Tensor& input,
                              torch::Tensor& output,
                              torch::Tensor& output_scale,
                              const torch::Tensor& masked_m,
                              int64_t group_size, double swiglu_limit) {
  const int64_t hidden = output.size(output.dim() - 1);
  const int64_t num_groups = hidden / group_size;
  const dim3 block(128);
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const float limit = static_cast<float>(swiglu_limit);

  if (input.dim() == 2) {
    const int64_t rows = input.size(0);
    const dim3 grid(static_cast<unsigned int>(rows * num_groups));
    swiglu_post_quant_2d_kernel<InT>
        <<<grid, block, block.x * sizeof(float), stream>>>(
            static_cast<const InT*>(input.data_ptr()), output.data_ptr(),
            output_scale.data_ptr(),
            static_cast<const int64_t*>(masked_m.data_ptr()), rows, hidden,
            group_size, num_groups, limit);
  } else {
    const int64_t experts = input.size(0);
    const int64_t rows = input.size(1);
    const dim3 grid(
        static_cast<unsigned int>(experts * rows * num_groups));
    swiglu_post_quant_3d_kernel<InT>
        <<<grid, block, block.x * sizeof(float), stream>>>(
            static_cast<const InT*>(input.data_ptr()), output.data_ptr(),
            output_scale.data_ptr(),
            static_cast<const int64_t*>(masked_m.data_ptr()), experts, rows,
            hidden, group_size, num_groups, limit);
  }
}

}  // namespace

void deepseek_v4_mega_moe_pre_dispatch(
    const torch::Tensor& x, const torch::Tensor& topk_idx,
    const torch::Tensor& topk_weights, torch::Tensor& buf_x,
    torch::Tensor& buf_x_sf, torch::Tensor& buf_topk_idx,
    torch::Tensor& buf_topk_weights, int64_t quant_group_size) {
  TORCH_CHECK(x.dim() == 2, "x must be [num_tokens, hidden]");
  TORCH_CHECK(topk_idx.dim() == 2 && topk_weights.dim() == 2,
              "topk tensors must be 2D");
  TORCH_CHECK(topk_idx.sizes() == topk_weights.sizes(),
              "topk_idx and topk_weights must have matching shapes");
  TORCH_CHECK(topk_idx.size(0) == x.size(0),
              "topk tensors must match x rows");
  TORCH_CHECK(topk_idx.device() == x.device() &&
                  topk_weights.device() == x.device() &&
                  buf_x.device() == x.device() &&
                  buf_x_sf.device() == x.device() &&
                  buf_topk_idx.device() == x.device() &&
                  buf_topk_weights.device() == x.device(),
              "all tensors must be on the same device as x");
  TORCH_CHECK(buf_x.dim() == 2 && buf_x.size(0) >= x.size(0) &&
                  buf_x.size(1) == x.size(1),
              "buf_x must be [padded_max, hidden]");
  TORCH_CHECK(buf_topk_idx.sizes() == buf_topk_weights.sizes(),
              "topk output buffers must match");
  TORCH_CHECK(buf_topk_idx.dim() == 2 && buf_topk_idx.size(0) == buf_x.size(0) &&
                  buf_topk_idx.size(1) == topk_idx.size(1),
              "topk output buffers must be [padded_max, top_k]");
  TORCH_CHECK(quant_group_size == 32 || quant_group_size == 64 ||
                  quant_group_size == 128,
              "quant_group_size must be 32, 64, or 128");
  TORCH_CHECK(x.size(1) % quant_group_size == 0,
              "hidden must be divisible by quant_group_size");
  TORCH_CHECK(x.is_contiguous() && topk_idx.is_contiguous() &&
                  topk_weights.is_contiguous() && buf_x.is_contiguous() &&
                  buf_x_sf.is_contiguous() && buf_topk_idx.is_contiguous() &&
                  buf_topk_weights.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(buf_x.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "buf_x must be float8_e4m3fn");
  TORCH_CHECK(buf_x_sf.scalar_type() == torch::kInt32 ||
                  buf_x_sf.scalar_type() == torch::kUInt8,
              "buf_x_sf must be int32-packed or uint8 UE8M0 scales");
  TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32 &&
                  buf_topk_weights.scalar_type() == torch::kFloat32,
              "topk weights must be float32");
  TORCH_CHECK(topk_idx.scalar_type() == torch::kInt32 ||
                  topk_idx.scalar_type() == torch::kInt64,
              "topk_idx must be int32 or int64");
  TORCH_CHECK(buf_topk_idx.scalar_type() == torch::kInt32 ||
                  buf_topk_idx.scalar_type() == torch::kInt64,
              "buf_topk_idx must be int32 or int64");

  const int64_t num_groups = x.size(1) / quant_group_size;
  if (buf_x_sf.scalar_type() == torch::kInt32) {
    TORCH_CHECK(num_groups % 4 == 0,
                "int32 UE8M0 scale buffer requires groups % 4 == 0");
    TORCH_CHECK(buf_x_sf.numel() * 4 >= x.size(0) * num_groups,
                "buf_x_sf is too small for packed UE8M0 scales");
  } else {
    TORCH_CHECK(buf_x_sf.numel() >= x.size(0) * num_groups,
                "buf_x_sf is too small for UE8M0 scales");
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(x));
  if (x.scalar_type() == torch::kBFloat16) {
    launch_mega_moe_quant<__mt_bfloat16>(x, buf_x, buf_x_sf,
                                         quant_group_size);
  } else if (x.scalar_type() == torch::kFloat16) {
    launch_mega_moe_quant<__half>(x, buf_x, buf_x_sf, quant_group_size);
  } else {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32,
                "x dtype must be float32, float16, or bfloat16");
    launch_mega_moe_quant<float>(x, buf_x, buf_x_sf, quant_group_size);
  }

  if (topk_idx.scalar_type() == torch::kInt32) {
    launch_topk_copy<int32_t>(topk_idx, topk_weights, buf_topk_idx,
                              buf_topk_weights);
  } else {
    launch_topk_copy<int64_t>(topk_idx, topk_weights, buf_topk_idx,
                              buf_topk_weights);
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_mega_moe_pre_dispatch launch failed: ",
              musaGetErrorString(err));
}

void deepseek_v4_silu_and_mul_masked_post_quant(
    const torch::Tensor& input, torch::Tensor& output,
    torch::Tensor& output_scale, const torch::Tensor& masked_m,
    int64_t quant_group_size, double swiglu_limit) {
  TORCH_CHECK(input.dim() == 2 || input.dim() == 3,
              "input must be 2D or 3D");
  TORCH_CHECK(input.size(input.dim() - 1) == output.size(output.dim() - 1) * 2,
              "input last dimension must be twice output last dimension");
  TORCH_CHECK(input.dim() == output.dim(),
              "input/output rank must match");
  TORCH_CHECK(output.device() == input.device() &&
                  output_scale.device() == input.device() &&
                  masked_m.device() == input.device(),
              "all tensors must be on the same device as input");
  TORCH_CHECK(input.dim() == 2 ||
                  (input.size(0) == output.size(0) &&
                   input.size(1) == output.size(1)),
              "3D input/output leading dimensions must match");
  TORCH_CHECK(input.dim() == 3 ||
                  input.size(0) == output.size(0),
              "2D input/output rows must match");
  TORCH_CHECK(quant_group_size == 32 || quant_group_size == 64 ||
                  quant_group_size == 128,
              "quant_group_size must be 32, 64, or 128");
  TORCH_CHECK(output.size(output.dim() - 1) % quant_group_size == 0,
              "output hidden must be divisible by quant_group_size");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous() &&
                  output_scale.is_contiguous() && masked_m.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "output must be float8_e4m3fn");
  TORCH_CHECK(output_scale.scalar_type() == torch::kInt32 ||
                  output_scale.scalar_type() == torch::kUInt8,
              "output_scale must be int32-packed or uint8 UE8M0 scales");
  TORCH_CHECK(masked_m.scalar_type() == torch::kInt64,
              "masked_m must be int64 for native provider");
  if (input.dim() == 2) {
    TORCH_CHECK(masked_m.numel() == 1, "2D input requires scalar masked_m");
  } else {
    TORCH_CHECK(masked_m.numel() == input.size(0),
                "3D input requires one masked_m entry per expert");
  }

  const int64_t hidden = output.size(output.dim() - 1);
  const int64_t num_groups = hidden / quant_group_size;
  const int64_t rows = output.numel() / hidden;
  if (output_scale.scalar_type() == torch::kInt32) {
    TORCH_CHECK(num_groups % 4 == 0,
                "int32 UE8M0 scale buffer requires groups % 4 == 0");
    TORCH_CHECK(output_scale.numel() * 4 >= rows * num_groups,
                "output_scale is too small for packed UE8M0 scales");
  } else {
    TORCH_CHECK(output_scale.numel() >= rows * num_groups,
                "output_scale is too small for UE8M0 scales");
  }

  const at::musa::OptionalMUSAGuard device_guard(device_of(input));
  if (input.scalar_type() == torch::kBFloat16) {
    launch_swiglu_post_quant<__mt_bfloat16>(
        input, output, output_scale, masked_m, quant_group_size, swiglu_limit);
  } else if (input.scalar_type() == torch::kFloat16) {
    launch_swiglu_post_quant<__half>(
        input, output, output_scale, masked_m, quant_group_size, swiglu_limit);
  } else {
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input dtype must be float32, float16, or bfloat16");
    launch_swiglu_post_quant<float>(
        input, output, output_scale, masked_m, quant_group_size, swiglu_limit);
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess,
              "deepseek_v4_silu_and_mul_masked_post_quant launch failed: ",
              musaGetErrorString(err));
}
