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

template <typename OutT>
__device__ __forceinline__ void store_dequant(OutT* out, int64_t idx,
                                              float value) {
  out[idx] = static_cast<OutT>(value);
}

template <>
__device__ __forceinline__ void store_dequant<__mt_bfloat16>(
    __mt_bfloat16* out, int64_t idx, float value) {
  out[idx] = __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ void store_dequant<__half>(__half* out,
                                                      int64_t idx,
                                                      float value) {
  out[idx] = __float2half_rn(value);
}

template <typename OutT>
__global__ void mxfp4_dequant_kernel(const uint8_t* __restrict__ x,
                                     const uint8_t* __restrict__ scale,
                                     OutT* __restrict__ out,
                                     int64_t packed_numel) {
  const int64_t packed_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (packed_idx >= packed_numel) {
    return;
  }

  const uint8_t packed = x[packed_idx];
  const int64_t out_idx = packed_idx * 2;
  const uint8_t scale0 = scale[out_idx / 32];
  const uint8_t scale1 = scale[(out_idx + 1) / 32];
  const float value0 =
      mxfp4_e2m1_value(packed & 0x0f) * e8m0_scale_to_float(scale0);
  const float value1 =
      mxfp4_e2m1_value((packed >> 4) & 0x0f) *
      e8m0_scale_to_float(scale1);

  store_dequant(out, out_idx, value0);
  store_dequant(out, out_idx + 1, value1);
}

}  // namespace

void mxfp4_dequant(const torch::Tensor& x, const torch::Tensor& scale,
                   torch::Tensor& output) {
  TORCH_CHECK(x.scalar_type() == torch::kUInt8, "x must be uint8");
  TORCH_CHECK(scale.scalar_type() == torch::kUInt8, "scale must be uint8");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(scale.is_contiguous(), "scale must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(scale.device() == x.device() && output.device() == x.device(),
              "scale and output must be on the same device as x");
  TORCH_CHECK(output.numel() == x.numel() * 2,
              "output must have x.numel() * 2 elements");
  TORCH_CHECK(output.numel() % 32 == 0,
              "output element count must be divisible by 32");
  TORCH_CHECK(scale.numel() == output.numel() / 32,
              "scale must have one E8M0 byte per 32 output elements");
  TORCH_CHECK(output.scalar_type() == torch::kFloat32 ||
                  output.scalar_type() == torch::kFloat16 ||
                  output.scalar_type() == torch::kBFloat16,
              "output dtype must be float32, float16, or bfloat16");

  const at::musa::OptionalMUSAGuard device_guard(device_of(x));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  const int threads = 256;
  const int64_t packed_numel = x.numel();
  const dim3 block(threads);
  const dim3 grid(
      static_cast<unsigned int>((packed_numel + threads - 1) / threads));

  if (output.scalar_type() == torch::kFloat32) {
    mxfp4_dequant_kernel<float><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(scale.data_ptr()),
        static_cast<float*>(output.data_ptr()), packed_numel);
  } else if (output.scalar_type() == torch::kBFloat16) {
    mxfp4_dequant_kernel<__mt_bfloat16><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<__mt_bfloat16*>(output.data_ptr()), packed_numel);
  } else {
    mxfp4_dequant_kernel<__half><<<grid, block, 0, stream>>>(
        static_cast<const uint8_t*>(x.data_ptr()),
        static_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<__half*>(output.data_ptr()), packed_numel);
  }

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "mxfp4_dequant launch failed: ",
              musaGetErrorString(err));
}
