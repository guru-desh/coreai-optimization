# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Torch-native naive (no-SMAWK) O(n * k^2) 1-D k-means DP.

Experimental alternative to ``core.cluster(..., vectorize=True)`` (which runs the
same naive DP via a JIT-compiled C++ extension). This module operates on torch
tensors directly, on whatever device the caller puts them on (including CUDA), so
the actual O(n^2) compute can run on GPU. It does not go through the ctypes/C
extension ABI at all — a different implementation of the same algorithm, kept
alongside ``core.py`` for a controlled A/B comparison, not a replacement.

The production caller (``_KMeansFakePalettize._cluster_weights_1d``) moves weights
to CPU before this code ever runs (for reasons unrelated to kmeans1d — the
multiprocessing ``num_workers>1`` path uses ``spawn`` context, which cannot carry
CUDA tensors across the fork). So there is no "existing GPU residency" to
preserve; :func:`cluster` deliberately moves its (already deduplicated, so small)
input to ``device`` for the DP step and returns tensors on that same device,
accepting the transfer cost as the price of running the O(n^2) step on GPU.
"""

from __future__ import annotations

from collections import namedtuple

import torch

Clustered = namedtuple("Clustered", "clusters centroids")

_INF = float("inf")


def cluster(
    array: torch.Tensor,
    k: int,
    *,
    weights: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Clustered:
    """Cluster 1-D data into k clusters with the naive, torch-vectorized O(n * k^2) DP.

    :param array: 1-D tensor of values to cluster, in any order, on any device.
    :param k: Number of clusters (int).
    :param weights: Optional 1-D tensor of per-point weights, same length as `array`.
    :param device: Device to run the DP on. Defaults to CUDA if available, else CPU;
        `array`/`weights` are moved here regardless of their input device.
    :param dtype: Floating dtype to compute in. float32 by default; use float64 for
        tighter numerical parity with the SMAWK/C++ reference at some speed cost.
    :return: A ``Clustered(clusters, centroids)`` namedtuple of torch tensors
        (``clusters``: int64, ``centroids``: `dtype`), both on `device`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n = int(array.shape[0])
    assert k > 0, f"Invalid k: {k}"
    assert n > 0, f"Invalid len(array): {n}"
    k = min(k, n)

    values = array.to(device=device, dtype=dtype)
    sorted_values, order = torch.sort(values, stable=True)
    if weights is None:
        sorted_weights = torch.ones(n, dtype=dtype, device=device)
    else:
        sorted_weights = weights.to(device=device, dtype=dtype)[order]

    cumw, cumsum, cumsq = _prefix_sums(sorted_values, sorted_weights)
    cost_matrix = _pairwise_segment_costs(cumw, cumsum, cumsq)
    split_index = _run_dp(cost_matrix, n, k, dtype, device)
    centroids, sorted_labels = _backtrack(cumw, cumsum, split_index, n, k)

    labels = torch.empty(n, dtype=torch.int64, device=device)
    labels[order] = sorted_labels
    return Clustered(clusters=labels, centroids=centroids)


def _prefix_sums(
    sorted_values: torch.Tensor, sorted_weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 1-based prefix sums of weight, weight*value, and weight*value^2."""
    zero = sorted_values.new_zeros(1)
    cumw = torch.cat([zero, torch.cumsum(sorted_weights, dim=0)])
    cumsum = torch.cat([zero, torch.cumsum(sorted_weights * sorted_values, dim=0)])
    cumsq = torch.cat([zero, torch.cumsum(sorted_weights * sorted_values * sorted_values, dim=0)])
    return cumw, cumsum, cumsq


def _pairwise_segment_costs(
    cumw: torch.Tensor, cumsum: torch.Tensor, cumsq: torch.Tensor
) -> torch.Tensor:
    """Return an ``(n, n)`` matrix where ``[i - 1, j]`` is the weighted SSE of the
    segment spanning sorted points ``j+1..i`` (1-based, inclusive), for every
    ``i`` in ``1..n`` and ``j`` in ``0..n-1``; ``inf`` where ``j >= i`` (empty/invalid).
    """
    seg_w = cumw[1:].unsqueeze(1) - cumw[:-1].unsqueeze(0)
    seg_sum = cumsum[1:].unsqueeze(1) - cumsum[:-1].unsqueeze(0)
    seg_sq = cumsq[1:].unsqueeze(1) - cumsq[:-1].unsqueeze(0)
    valid = seg_w > 0
    safe_w = torch.where(valid, seg_w, torch.ones_like(seg_w))
    cost = seg_sq - seg_sum * seg_sum / safe_w
    inf = torch.full_like(cost, _INF)
    return torch.where(valid, cost, inf)


def _run_dp(
    cost_matrix: torch.Tensor, n: int, k: int, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Fill the DP table layer by layer and return the split-index table.

    For each layer (cluster count), ``D[m][i]`` (cost of clustering the first
    ``i`` sorted points into ``m`` clusters) is reduced via a single vectorized
    ``torch.min`` over the shared pairwise-cost matrix instead of SMAWK's
    monotonicity-exploiting search — the naive O(n) per row / O(n^2) per layer
    reduction, run as one array op per layer instead of a Python-level loop.

    :return: ``(k + 1, n)`` int64 tensor where ``split_index[m, i - 1]`` is the
        optimal 0-based split point when clustering the first ``i`` sorted points
        into ``m`` clusters.
    """
    split_index = torch.zeros((k + 1, n), dtype=torch.int64, device=device)
    cost_prev = torch.full((n,), _INF, dtype=dtype, device=device)
    cost_prev[0] = 0.0
    for clusters in range(1, k + 1):
        candidate = cost_matrix + cost_prev.unsqueeze(0)
        cost_cur, argmin = torch.min(candidate, dim=1)
        split_index[clusters] = argmin
        inf_tail = cost_prev.new_full((1,), _INF)
        cost_prev = torch.cat([inf_tail, cost_cur[:-1]])
    return split_index


def _backtrack(
    cumw: torch.Tensor,
    cumsum: torch.Tensor,
    split_index: torch.Tensor,
    n: int,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover ascending centroids and sorted-order labels from the split table."""
    device = cumw.device
    centroids = torch.empty(k, dtype=cumw.dtype, device=device)
    sorted_labels = torch.empty(n, dtype=torch.int64, device=device)
    end = n
    for clusters in range(k, 0, -1):
        start = int(split_index[clusters, end - 1].item()) + 1
        weight_total = cumw[end] - cumw[start - 1]
        centroids[clusters - 1] = (cumsum[end] - cumsum[start - 1]) / weight_total
        sorted_labels[start - 1 : end] = clusters - 1
        end = start - 1
    return centroids, sorted_labels
