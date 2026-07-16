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
    dtype: str = "float64",
) -> Clustered:
    """
    :param array: A sequence of floats
    :param k: Number of clusters (int)
    :param weights: Sequence of weights (if provided, must have same length as `array`)
    :param dtype: Either "float64" (default, matches the original vendored precision) or
        "float32" (halves the C-side scalar width; output will not exactly match "float64").
    :return: A tuple with (clusters, centroids)
    """
    assert dtype in ("float64", "float32"), f"Invalid dtype: {dtype}"
    assert k > 0, f"Invalid k: {k}"
    n = len(array)
    assert n > 0, f"Invalid len(array): {n}"
    k = min(k, n)

    if weights is not None:
        assert len(weights) == n, f"len(weights)={len(weights)} != len(array)={n}"

    c_scalar = ctypes.c_float if dtype == "float32" else ctypes.c_double
    c_array = (c_scalar * n)(*array)
    c_n = ctypes.c_ulong(n)
    c_k = ctypes.c_ulong(k)
    c_clusters = (ctypes.c_ulong * n)()
    c_centroids = (c_scalar * k)()

    if weights is None:
        fn = _dll().cluster_f32 if dtype == "float32" else _dll().cluster
        fn(c_array, c_n, c_k, c_clusters, c_centroids)
    else:
        c_weights = (c_scalar * n)(*weights)
        fn = _dll().cluster_with_weights_f32 if dtype == "float32" else _dll().cluster_with_weights
        fn(c_array, c_weights, c_n, c_k, c_clusters, c_centroids)

    clusters = list(c_clusters)
    centroids = list(c_centroids)

    output = Clustered(clusters=clusters, centroids=centroids)

    return output
