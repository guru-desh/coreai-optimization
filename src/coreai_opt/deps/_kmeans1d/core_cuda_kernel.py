# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Python wrapper for the hand-written CUDA kernel in ``_cuda_kernel.cu``.

Same naive (no-SMAWK) O(n * k^2) DP as ``core.cluster(..., vectorize=True)``
(C++) and ``core_torch.cluster()`` (torch tensor ops), a third implementation
of the identical algorithm. Built to fix a measured problem with
``core_torch.cluster()``: its backtrack step syncs with the host once per
cluster (``.item()`` on a CUDA tensor, k times per weight block). This
kernel's entire forward DP pass and backtrack run as a fixed sequence of
kernel launches queued on the default CUDA stream from C++ — see
``_cuda_kernel.cu``'s module docstring for the full design rationale.

Requires CUDA: unlike ``core_torch.cluster()``, there is no CPU fallback —
the kernel source uses ``__global__``/``<<<...>>>`` launch syntax that only
compiles/runs against an actual CUDA device.
"""

from __future__ import annotations

import ctypes
import os
from collections import namedtuple

import torch
from torch.utils.cpp_extension import load

Clustered = namedtuple("Clustered", "clusters centroids")

_DLL = None


def _dll():
    global _DLL
    if _DLL is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "core_cuda_kernel.cluster() requires CUDA; no CUDA device is available."
            )
        _extension = load(
            name="_cuda_kernel",
            sources=[os.path.join(os.path.dirname(__file__), "_cuda_kernel.cu")],
            extra_cuda_cflags=["-O2"],
        )
        _DLL = ctypes.cdll.LoadLibrary(_extension.__file__)
    return _DLL


def cluster(
    array: torch.Tensor,
    k: int,
    *,
    weights: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> Clustered:
    """Cluster 1-D data into k clusters with the naive, CUDA-kernel O(n * k^2) DP.

    :param array: 1-D tensor of values to cluster, in any order. Moved to CUDA
        if not already there.
    :param k: Number of clusters (int).
    :param weights: Optional 1-D tensor of per-point weights, same length as
        `array`. `None` is equivalent to all-ones (unweighted) — the kernel
        itself only has a single, always-weighted code path.
    :param dtype: Floating dtype to compute in. float64 by default, matching
        the SMAWK/C++ reference's precision — the torch-tensor-ops backend
        (`core_torch.py`) was found to silently diverge from that reference
        at float32 (matching structural but not full MLIR export hashes), so
        this backend defaults to the precision known to be exact instead of
        repeating that mistake.
    :return: A ``Clustered(clusters, centroids)`` namedtuple of CUDA tensors
        (``clusters``: int64, ``centroids``: `dtype`).
    """
    n = int(array.shape[0])
    assert k > 0, f"Invalid k: {k}"
    assert n > 0, f"Invalid len(array): {n}"
    k = min(k, n)

    if dtype != torch.float64:
        raise ValueError(
            f"_cuda_kernel.cu's launcher takes `double*` pointers; dtype must be "
            f"torch.float64, got {dtype}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("core_cuda_kernel.cluster() requires CUDA; no CUDA device is available.")

    device = "cuda"
    values = array.to(device=device, dtype=dtype)
    sorted_values, order = torch.sort(values, stable=True)
    sorted_weights = (
        torch.ones(n, dtype=dtype, device=device)
        if weights is None
        else weights.to(device=device, dtype=dtype)[order]
    )

    cumw = torch.empty(n + 1, dtype=dtype, device=device)
    cumsum = torch.empty(n + 1, dtype=dtype, device=device)
    cumsum2 = torch.empty(n + 1, dtype=dtype, device=device)
    buf_a = torch.empty(n + 1, dtype=dtype, device=device)
    buf_b = torch.empty(n + 1, dtype=dtype, device=device)
    split_index = torch.empty((k, n), dtype=torch.int64, device=device)
    centroids = torch.empty(k, dtype=dtype, device=device)
    sorted_labels = torch.empty(n, dtype=torch.int64, device=device)

    _dll().cluster_naive_dp_cuda(
        ctypes.c_void_p(sorted_values.data_ptr()),
        ctypes.c_void_p(sorted_weights.data_ptr()),
        ctypes.c_void_p(cumw.data_ptr()),
        ctypes.c_void_p(cumsum.data_ptr()),
        ctypes.c_void_p(cumsum2.data_ptr()),
        ctypes.c_void_p(buf_a.data_ptr()),
        ctypes.c_void_p(buf_b.data_ptr()),
        ctypes.c_void_p(split_index.data_ptr()),
        ctypes.c_void_p(centroids.data_ptr()),
        ctypes.c_void_p(sorted_labels.data_ptr()),
        ctypes.c_ulong(n),
        ctypes.c_ulong(k),
    )

    labels = torch.empty(n, dtype=torch.int64, device=device)
    labels[order] = sorted_labels
    return Clustered(clusters=labels, centroids=centroids)
