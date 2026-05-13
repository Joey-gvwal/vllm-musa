// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#ifndef VLLM_MUSA_TVM_FFI_EXTRA_MUSA_DEVICE_GUARD_H_
#define VLLM_MUSA_TVM_FFI_EXTRA_MUSA_DEVICE_GUARD_H_

#include <musa_runtime.h>
#include <tvm/ffi/error.h>

namespace tvm {
namespace ffi {

#define TVM_FFI_CHECK_MUSA_ERROR(stmt)                                            \
  do {                                                                            \
    musaError_t __err = (stmt);                                                   \
    if (__err != musaSuccess) {                                                   \
      TVM_FFI_THROW(RuntimeError)                                                 \
          << "MUSA Runtime Error (" << static_cast<int>(__err)                    \
          << "): " << musaGetErrorString(__err);                                  \
    }                                                                             \
  } while (0)

struct MUSADeviceGuard {
  MUSADeviceGuard() = delete;

  explicit MUSADeviceGuard(int device_index) {
    target_device_index_ = device_index;
    TVM_FFI_CHECK_MUSA_ERROR(musaGetDevice(&original_device_index_));
    if (target_device_index_ != original_device_index_) {
      TVM_FFI_CHECK_MUSA_ERROR(musaSetDevice(device_index));
    }
  }

  ~MUSADeviceGuard() noexcept(false) {
    if (original_device_index_ != target_device_index_) {
      TVM_FFI_CHECK_MUSA_ERROR(musaSetDevice(original_device_index_));
    }
  }

 private:
  int original_device_index_;
  int target_device_index_;
};

}  // namespace ffi
}  // namespace tvm

#endif  // VLLM_MUSA_TVM_FFI_EXTRA_MUSA_DEVICE_GUARD_H_
