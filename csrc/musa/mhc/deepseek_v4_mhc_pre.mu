#include <cmath>
#include <cstdint>

#include <musa_bf16.h>
#include <musa_runtime.h>
#include <torch/all.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace {

constexpr int kHcMult = 4;
constexpr int kMixCount = kHcMult * 2 + kHcMult * kHcMult;
constexpr int kThreads = 256;

void check_musa_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_privateuseone(), name,
              " must be a MUSA tensor");
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__global__ void deepseek_v4_mhc_pre_kernel(
    const __mt_bfloat16* __restrict__ residual,
    const float* __restrict__ fn, const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base, float* __restrict__ post_mix,
    float* __restrict__ comb_mix, __mt_bfloat16* __restrict__ layer_input,
    int64_t num_tokens, int64_t hidden_size, float rms_eps, float hc_pre_eps,
    float hc_sinkhorn_eps, float hc_post_mult_value, int sinkhorn_repeat) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);
  if (token >= num_tokens) {
    return;
  }

  const int tid = threadIdx.x;
  const int64_t hc_hidden_size = kHcMult * hidden_size;
  const __mt_bfloat16* token_residual = residual + token * hc_hidden_size;

  float local_mix[kMixCount];
#pragma unroll
  for (int i = 0; i < kMixCount; ++i) {
    local_mix[i] = 0.0f;
  }
  float local_sqrsum = 0.0f;

  for (int64_t col = tid; col < hc_hidden_size; col += blockDim.x) {
    const float value = __bfloat162float(token_residual[col]);
    local_sqrsum += value * value;
#pragma unroll
    for (int mix = 0; mix < kMixCount; ++mix) {
      local_mix[mix] += value * fn[mix * hc_hidden_size + col];
    }
  }

  __shared__ float mix_reduce[kMixCount][kThreads];
  __shared__ float sq_reduce[kThreads];
  __shared__ float pre_shared[kHcMult];

#pragma unroll
  for (int mix = 0; mix < kMixCount; ++mix) {
    mix_reduce[mix][tid] = local_mix[mix];
  }
  sq_reduce[tid] = local_sqrsum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int mix = 0; mix < kMixCount; ++mix) {
        mix_reduce[mix][tid] += mix_reduce[mix][tid + stride];
      }
      sq_reduce[tid] += sq_reduce[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    float mixes[kMixCount];
    const float inv_rms =
        rsqrtf(sq_reduce[0] / static_cast<float>(hc_hidden_size) + rms_eps);
#pragma unroll
    for (int mix = 0; mix < kMixCount; ++mix) {
      mixes[mix] = mix_reduce[mix][0] * inv_rms;
    }

#pragma unroll
    for (int i = 0; i < kHcMult; ++i) {
      const float pre =
          sigmoidf_fast(mixes[i] * hc_scale[0] + hc_base[i]) + hc_pre_eps;
      pre_shared[i] = pre;
      post_mix[token * kHcMult + i] =
          sigmoidf_fast(mixes[kHcMult + i] * hc_scale[1] +
                        hc_base[kHcMult + i]) *
          hc_post_mult_value;
    }

    float comb[kHcMult][kHcMult];
#pragma unroll
    for (int row = 0; row < kHcMult; ++row) {
      float row_max = -INFINITY;
#pragma unroll
      for (int col = 0; col < kHcMult; ++col) {
        const int idx = 2 * kHcMult + row * kHcMult + col;
        const float value = mixes[idx] * hc_scale[2] + hc_base[idx];
        comb[row][col] = value;
        row_max = fmaxf(row_max, value);
      }
      float row_sum = 0.0f;
#pragma unroll
      for (int col = 0; col < kHcMult; ++col) {
        const float value = expf(comb[row][col] - row_max);
        comb[row][col] = value;
        row_sum += value;
      }
#pragma unroll
      for (int col = 0; col < kHcMult; ++col) {
        comb[row][col] = comb[row][col] / row_sum + hc_sinkhorn_eps;
      }
    }

#pragma unroll
    for (int col = 0; col < kHcMult; ++col) {
      float col_sum = 0.0f;
#pragma unroll
      for (int row = 0; row < kHcMult; ++row) {
        col_sum += comb[row][col];
      }
#pragma unroll
      for (int row = 0; row < kHcMult; ++row) {
        comb[row][col] /= col_sum + hc_sinkhorn_eps;
      }
    }

    for (int repeat = 1; repeat < sinkhorn_repeat; ++repeat) {
#pragma unroll
      for (int row = 0; row < kHcMult; ++row) {
        float row_sum = 0.0f;
#pragma unroll
        for (int col = 0; col < kHcMult; ++col) {
          row_sum += comb[row][col];
        }
#pragma unroll
        for (int col = 0; col < kHcMult; ++col) {
          comb[row][col] /= row_sum + hc_sinkhorn_eps;
        }
      }
#pragma unroll
      for (int col = 0; col < kHcMult; ++col) {
        float col_sum = 0.0f;
#pragma unroll
        for (int row = 0; row < kHcMult; ++row) {
          col_sum += comb[row][col];
        }
#pragma unroll
        for (int row = 0; row < kHcMult; ++row) {
          comb[row][col] /= col_sum + hc_sinkhorn_eps;
        }
      }
    }

#pragma unroll
    for (int row = 0; row < kHcMult; ++row) {
#pragma unroll
      for (int col = 0; col < kHcMult; ++col) {
        comb_mix[token * kHcMult * kHcMult + row * kHcMult + col] =
            comb[row][col];
      }
    }
  }
  __syncthreads();

  for (int64_t h = tid; h < hidden_size; h += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int hc = 0; hc < kHcMult; ++hc) {
      value += pre_shared[hc] *
               __bfloat162float(token_residual[hc * hidden_size + h]);
    }
    layer_input[token * hidden_size + h] = __float2bfloat16(value);
  }
}

}  // namespace

void deepseek_v4_mhc_pre(
    const torch::Tensor& residual, const torch::Tensor& fn,
    const torch::Tensor& hc_scale, const torch::Tensor& hc_base,
    torch::Tensor& post_mix, torch::Tensor& comb_mix,
    torch::Tensor& layer_input, double rms_eps, double hc_pre_eps,
    double hc_sinkhorn_eps, double hc_post_mult_value,
    int64_t sinkhorn_repeat) {
  check_musa_tensor(residual, "residual");
  check_musa_tensor(fn, "fn");
  check_musa_tensor(hc_scale, "hc_scale");
  check_musa_tensor(hc_base, "hc_base");
  check_musa_tensor(post_mix, "post_mix");
  check_musa_tensor(comb_mix, "comb_mix");
  check_musa_tensor(layer_input, "layer_input");
  TORCH_CHECK(residual.device() == fn.device(), "fn device mismatch");
  TORCH_CHECK(residual.device() == hc_scale.device(), "hc_scale device mismatch");
  TORCH_CHECK(residual.device() == hc_base.device(), "hc_base device mismatch");
  TORCH_CHECK(residual.device() == post_mix.device(), "post_mix device mismatch");
  TORCH_CHECK(residual.device() == comb_mix.device(), "comb_mix device mismatch");
  TORCH_CHECK(residual.device() == layer_input.device(),
              "layer_input device mismatch");

  TORCH_CHECK(residual.scalar_type() == torch::kBFloat16,
              "residual must be bfloat16");
  TORCH_CHECK(fn.scalar_type() == torch::kFloat32, "fn must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == torch::kFloat32,
              "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == torch::kFloat32,
              "hc_base must be float32");
  TORCH_CHECK(post_mix.scalar_type() == torch::kFloat32,
              "post_mix must be float32");
  TORCH_CHECK(comb_mix.scalar_type() == torch::kFloat32,
              "comb_mix must be float32");
  TORCH_CHECK(layer_input.scalar_type() == torch::kBFloat16,
              "layer_input must be bfloat16");

  TORCH_CHECK(residual.dim() == 3, "residual must have shape [T, 4, H]");
  TORCH_CHECK(residual.size(1) == kHcMult, "residual hc dimension must be 4");
  TORCH_CHECK(fn.dim() == 2, "fn must have shape [24, 4 * H]");
  TORCH_CHECK(fn.size(0) == kMixCount, "fn first dimension must be 24");
  TORCH_CHECK(fn.size(1) == residual.size(1) * residual.size(2),
              "fn second dimension must equal 4 * hidden_size");
  TORCH_CHECK(hc_scale.sizes() == torch::IntArrayRef({3}),
              "hc_scale must have shape [3]");
  TORCH_CHECK(hc_base.sizes() == torch::IntArrayRef({kMixCount}),
              "hc_base must have shape [24]");
  TORCH_CHECK(post_mix.sizes() ==
                  torch::IntArrayRef({residual.size(0), kHcMult}),
              "post_mix must have shape [T, 4]");
  TORCH_CHECK(comb_mix.sizes() ==
                  torch::IntArrayRef({residual.size(0), kHcMult, kHcMult}),
              "comb_mix must have shape [T, 4, 4]");
  TORCH_CHECK(layer_input.sizes() ==
                  torch::IntArrayRef({residual.size(0), residual.size(2)}),
              "layer_input must have shape [T, H]");

  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(fn.is_contiguous(), "fn must be contiguous");
  TORCH_CHECK(hc_scale.is_contiguous(), "hc_scale must be contiguous");
  TORCH_CHECK(hc_base.is_contiguous(), "hc_base must be contiguous");
  TORCH_CHECK(post_mix.is_contiguous(), "post_mix must be contiguous");
  TORCH_CHECK(comb_mix.is_contiguous(), "comb_mix must be contiguous");
  TORCH_CHECK(layer_input.is_contiguous(), "layer_input must be contiguous");

  if (residual.size(0) == 0) {
    return;
  }
  TORCH_CHECK(sinkhorn_repeat >= 1, "sinkhorn_repeat must be >= 1");

  const at::musa::OptionalMUSAGuard device_guard(device_of(residual));
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  deepseek_v4_mhc_pre_kernel<<<residual.size(0), kThreads, 0, stream>>>(
      static_cast<const __mt_bfloat16*>(residual.data_ptr()),
      static_cast<const float*>(fn.data_ptr()),
      static_cast<const float*>(hc_scale.data_ptr()),
      static_cast<const float*>(hc_base.data_ptr()),
      static_cast<float*>(post_mix.data_ptr()),
      static_cast<float*>(comb_mix.data_ptr()),
      static_cast<__mt_bfloat16*>(layer_input.data_ptr()), residual.size(0),
      residual.size(2), static_cast<float>(rms_eps),
      static_cast<float>(hc_pre_eps), static_cast<float>(hc_sinkhorn_eps),
      static_cast<float>(hc_post_mult_value),
      static_cast<int>(sinkhorn_repeat));

  const auto err = musaGetLastError();
  TORCH_CHECK(err == musaSuccess, "deepseek_v4_mhc_pre launch failed: ",
              musaGetErrorString(err));
}
