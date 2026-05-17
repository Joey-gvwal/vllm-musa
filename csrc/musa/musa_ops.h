#include <optional>
#include <torch/library.h>

#include "core/scalar_type.hpp"

#include <vector>

void musa_fused_gemv_moe(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    torch::Tensor &topk_weights,
    torch::Tensor &topk_ids,
    bool mul_routed_weight,
    int64_t topk,
    bool use_int4_w4a16,
    bool use_swigelu);

void musa_fused_gemv(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &C,
    const c10::optional<torch::Tensor> &A_scale,
    const c10::optional<torch::Tensor> &B_scale,
    bool use_int4_w4a16,
    bool use_swigelu,
    bool use_rms_norm,
    const c10::optional<torch::Tensor> &gamma,
    double eps);

void per_token_group_quant_fp8(
    const torch::Tensor& input,
    torch::Tensor& output_q, torch::Tensor& output_s,
    int64_t group_size, double eps, double fp8_min,
    double fp8_max, bool scale_ue8m0,
    bool dummy_is_scale_transposed = false,
    bool dummy_is_tma_aligned = false);

void mxfp4_dequant(
    const torch::Tensor& x,
    const torch::Tensor& scale,
    torch::Tensor& output);

void mxfp4_grouped_gemv(
    const torch::Tensor& input,
    const torch::Tensor& packed_weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& expert_ids,
    torch::Tensor& output,
    const c10::optional<torch::Tensor>& expert_map);

void fp8_ds_mla_sparse_gather(
    const torch::Tensor& cache,
    const torch::Tensor& indices,
    const c10::optional<torch::Tensor>& lengths,
    torch::Tensor& output,
    torch::Tensor& valid);

void fused_deepseek_v4_qnorm_rope_kv_insert(
    torch::Tensor& q,
    const torch::Tensor& kv,
    torch::Tensor& k_cache,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& positions,
    const torch::Tensor& cos_sin_cache,
    double eps,
    int64_t block_size);
