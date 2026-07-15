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


class _ClusterItem(ctypes.Structure):
    """One cluster_batch() input item. Field order/types must match _core.cpp's ClusterItem."""

    _fields_ = [
        ("array", ctypes.POINTER(ctypes.c_double)),
        ("n", ctypes.c_ulong),
        ("k", ctypes.c_ulong),
        ("weights", ctypes.POINTER(ctypes.c_double)),
    ]


class _ClusterResult(ctypes.Structure):
    """One cluster_batch() output item. Field order/types must match _core.cpp's ClusterResult."""

    _fields_ = [
        ("clusters", ctypes.POINTER(ctypes.c_ulong)),
        ("centroids", ctypes.POINTER(ctypes.c_double)),
    ]


_EXTRA_CFLAGS = ["-std=c++11", "-O3", "-DNDEBUG"] + (
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


def cluster(array: Sequence[float], k: int, *, weights: Sequence[float] | None = None) -> Clustered:
    """
    :param array: A sequence of floats
    :param k: Number of clusters (int)
    :param weights: Sequence of weights (if provided, must have same length as `array`)
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
        _dll().cluster(c_array, c_n, c_k, c_clusters, c_centroids)
    else:
        c_weights = (ctypes.c_double * n)(*weights)
        _dll().cluster_with_weights(c_array, c_weights, c_n, c_k, c_clusters, c_centroids)

    clusters = list(c_clusters)
    centroids = list(c_centroids)

    output = Clustered(clusters=clusters, centroids=centroids)

    return output


def cluster_batch(
    items: Sequence[tuple[Sequence[float], int, Sequence[float] | None]],
) -> list[Clustered]:
    """
    Cluster multiple (array, k, weights) items in a single C call instead of
    one Python/ctypes round trip per item. Each item's result is
    bit-identical to calling `cluster()` on it individually.

    :param items: sequence of (array, k, weights) tuples, matching cluster()'s args
    :return: list of Clustered results, in the same order as `items`
    """
    num_items = len(items)
    c_items = (_ClusterItem * num_items)()
    c_results = (_ClusterResult * num_items)()

    # ctypes doesn't keep referenced buffers alive once assigned into a
    # struct's pointer field, so hold direct references until the call returns.
    c_arrays = []
    c_weights_arrays = []
    c_clusters_arrays = []
    c_centroids_arrays = []

    for idx, (array, k, weights) in enumerate(items):
        assert k > 0, f"Invalid k: {k}"
        n = len(array)
        assert n > 0, f"Invalid len(array): {n}"
        k = min(k, n)

        if weights is not None:
            assert len(weights) == n, f"len(weights)={len(weights)} != len(array)={n}"

        c_array = (ctypes.c_double * n)(*array)
        c_clusters = (ctypes.c_ulong * n)()
        c_centroids = (ctypes.c_double * k)()
        c_arrays.append(c_array)
        c_clusters_arrays.append(c_clusters)
        c_centroids_arrays.append(c_centroids)

        c_items[idx].array = ctypes.cast(c_array, ctypes.POINTER(ctypes.c_double))
        c_items[idx].n = n
        c_items[idx].k = k
        if weights is None:
            c_items[idx].weights = ctypes.POINTER(ctypes.c_double)()
        else:
            c_weights = (ctypes.c_double * n)(*weights)
            c_weights_arrays.append(c_weights)
            c_items[idx].weights = ctypes.cast(c_weights, ctypes.POINTER(ctypes.c_double))

        c_results[idx].clusters = ctypes.cast(c_clusters, ctypes.POINTER(ctypes.c_ulong))
        c_results[idx].centroids = ctypes.cast(c_centroids, ctypes.POINTER(ctypes.c_double))

    _dll().cluster_batch(c_items, ctypes.c_ulong(num_items), c_results)

    return [
        Clustered(clusters=list(c_clusters_arrays[idx]), centroids=list(c_centroids_arrays[idx]))
        for idx in range(num_items)
    ]
