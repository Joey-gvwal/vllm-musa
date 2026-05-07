# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

PATCHES = [
    (
        "(n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)",
        "((n_lo | (n_hi << 8)) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)",
    ),
    (
        "(sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)",
        "((sc_lo | (sc_hi << 8)) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)",
    ),
    (
        "(zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)",
        "((zr_lo | (zr_hi << 8)) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)",
    ),
]
