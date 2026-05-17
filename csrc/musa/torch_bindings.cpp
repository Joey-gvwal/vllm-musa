#include "cache.h"
#include "cuda_utils.h"
#include "musa_ops.h"
#include "core/registration.h"

#include <torch/library.h>
#include <torch/version.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"

TORCH_LIBRARY_EXPAND(CONCAT(TORCH_EXTENSION_NAME, _musa_ops), musa_ops) {
#ifdef USE_MUSA
  musa_ops.def(
      "musa_fused_gemv_moe(Tensor! A, Tensor! B, Tensor! C, Tensor? A_scale, Tensor? B_scale,"
      "Tensor! topk_weights, Tensor! topk_ids, bool mul_routed_weight, int topk, bool use_int4_w4a16,"
      "bool use_swigelu) -> ()");
  musa_ops.impl("musa_fused_gemv_moe", torch::kMUSA, &musa_fused_gemv_moe);

  musa_ops.def(
      "musa_fused_gemv(Tensor! A, Tensor! B, Tensor! C, Tensor? A_scale, Tensor? B_scale,"
      "bool use_int4_w4a16, bool use_swigelu, bool use_rms_norm, Tensor? gamma,"
      "float eps) -> ()");
  musa_ops.impl("musa_fused_gemv", torch::kMUSA, &musa_fused_gemv);

  musa_ops.def(
      "per_token_group_fp8_quant(Tensor input, Tensor! output_q, Tensor! "
      "output_s, "
      "int group_size, float eps, float fp8_min, float fp8_max, bool "
      "scale_ue8m0, bool dummy_is_scale_transposed, bool dummy_is_tma_aligned "
      ") -> ()");
  musa_ops.impl("per_token_group_fp8_quant", torch::kMUSA,
           &per_token_group_quant_fp8);

  musa_ops.def(
      "mxfp4_dequant(Tensor x, Tensor scale, Tensor! output) -> ()");
  musa_ops.impl("mxfp4_dequant", torch::kMUSA, &mxfp4_dequant);

  musa_ops.def(
      "mxfp4_grouped_gemv(Tensor input, Tensor packed_weight, "
      "Tensor weight_scale, Tensor expert_ids, Tensor! output, "
      "Tensor? expert_map) -> ()");
  musa_ops.impl("mxfp4_grouped_gemv", torch::kMUSA, &mxfp4_grouped_gemv);

  musa_ops.def(
      "deepseek_v4_mega_moe_pre_dispatch(Tensor x, Tensor topk_idx, "
      "Tensor topk_weights, Tensor! buf_x, Tensor! buf_x_sf, "
      "Tensor! buf_topk_idx, Tensor! buf_topk_weights, int "
      "quant_group_size) -> ()");
  musa_ops.impl("deepseek_v4_mega_moe_pre_dispatch", torch::kMUSA,
           &deepseek_v4_mega_moe_pre_dispatch);

  musa_ops.def(
      "deepseek_v4_silu_and_mul_masked_post_quant(Tensor input, "
      "Tensor! output, Tensor! output_scale, Tensor masked_m, int "
      "quant_group_size, float swiglu_limit) -> ()");
  musa_ops.impl("deepseek_v4_silu_and_mul_masked_post_quant", torch::kMUSA,
           &deepseek_v4_silu_and_mul_masked_post_quant);

  musa_ops.def(
      "fp8_ds_mla_sparse_gather(Tensor cache, Tensor indices, "
      "Tensor? lengths, Tensor! output, Tensor! valid) -> ()");
  musa_ops.impl("fp8_ds_mla_sparse_gather", torch::kMUSA,
           &fp8_ds_mla_sparse_gather);

  musa_ops.def(
      "fused_deepseek_v4_qnorm_rope_kv_insert(Tensor! q, Tensor kv, "
      "Tensor! k_cache, Tensor slot_mapping, Tensor positions, "
      "Tensor cos_sin_cache, float eps, int block_size) -> ()");
  musa_ops.impl("fused_deepseek_v4_qnorm_rope_kv_insert", torch::kMUSA,
           &fused_deepseek_v4_qnorm_rope_kv_insert);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
