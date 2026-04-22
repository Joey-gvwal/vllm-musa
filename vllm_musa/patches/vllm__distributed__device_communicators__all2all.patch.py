# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.distributed.device.communicators.all2all.

"""

PATCHES = [
    # Remove explicitly_destroy argument from all2all calls, as MUSA's version
    # of the communicator does not support it.
    ("explicitly_destroy=True,", ""),
]
