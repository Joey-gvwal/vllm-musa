# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Patch for vllm.distributed.device.communicators.all2all.

"""

PATCHES = [
    # Remove explicitly_destroy argument from all2all calls, as MUSA's version
    # of the communicator does not support it.
    ("explicitly_destroy=True,", ""),
    (
        """    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()
""",
        """    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                destroy = getattr(handle, "destroy", None)
                if callable(destroy):
                    destroy()
            self.handle_cache._cache.clear()
""",
    ),
]
