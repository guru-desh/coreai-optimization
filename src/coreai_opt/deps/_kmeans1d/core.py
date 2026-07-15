# MIT License
#
# Copyright (c) 2019 Daniel Steinberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright © 2026 Apple Inc.

import ctypes
import os
import sys
from collections import namedtuple
from collections.abc import Sequence

from torch.utils.cpp_extension import load

Clustered = namedtuple("Clustered", "clusters centroids")

_EXTRA_CFLAGS = ["-std=c++11", "-O2", "-DNDEBUG"] + (
    ["-stdlib=libc++"] if sys.platform == "darwin" else []
)
_EXTRA_LDFLAGS = ["-stdlib=libc++"] if sys.platform == "darwin" else []

_DLL = None


def _dll():
    global _DLL
    if _DLL is None:
        _core = load(
            name="_core",
            sources=[os.path.join(os.path.dirname(__file__), "_core.cpp")],
            extra_cflags=_EXTRA_CFLAGS,
            extra_ldflags=_EXTRA_LDFLAGS,
        )
        _DLL = ctypes.cdll.LoadLibrary(_core.__file__)
    return _DLL


def cluster(
    array: Sequence[float],
    k: int,
    *,
    weights: Sequence[float] | None = None,
    vectorize: bool = False,
) -> Clustered:
    """
    :param array: A sequence of floats
    :param k: Number of clusters (int)
    :param weights: Sequence of weights (if provided, must have same length as `array`)
    :param vectorize: If True, use a naive O(len(array) * k^2) dynamic program instead
        of the default SMAWK-accelerated O(len(array) * k) one. SMAWK and the naive
        DP compute the identical optimum (same D/T table; SMAWK is only a faster way
        to find the same row-minima), so results are equivalent either way. In
        practice the naive DP's flat, branch-free reduction can be faster than
        SMAWK's recursive stack search for small inputs (empirically, roughly
        len(array) <~ 200-300, shrinking as k grows), but becomes dramatically
        *slower* past that crossover (e.g. 24x slower was measured at
        len(array)=4217, k=256) since its time complexity is strictly worse. There
        is no automatic fallback or size guard: pass True only when len(array) is
        known to be small for the calling context.
    :return: A tuple with (clusters, centroids)
    """
    assert k > 0, f"Invalid k: {k}"
    n = len(array)
    assert n > 0, f"Invalid len(array): {n}"
    k = min(k, n)

    if weights is not None:
        assert len(weights) == n, f"len(weights)={len(weights)} != len(array)={n}"

    c_array = (ctypes.c_double * n)(*array)
    c_n = ctypes.c_ulong(n)
    c_k = ctypes.c_ulong(k)
    c_clusters = (ctypes.c_ulong * n)()
    c_centroids = (ctypes.c_double * k)()

    if weights is None:
        cluster_fn = _dll().cluster_vectorized if vectorize else _dll().cluster
        cluster_fn(c_array, c_n, c_k, c_clusters, c_centroids)
    else:
        c_weights = (ctypes.c_double * n)(*weights)
        cluster_with_weights_fn = (
            _dll().cluster_vectorized_with_weights if vectorize else _dll().cluster_with_weights
        )
        cluster_with_weights_fn(c_array, c_weights, c_n, c_k, c_clusters, c_centroids)

    clusters = list(c_clusters)
    centroids = list(c_centroids)

    output = Clustered(clusters=clusters, centroids=centroids)

    return output
