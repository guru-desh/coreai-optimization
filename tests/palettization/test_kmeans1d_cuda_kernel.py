# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Correctness tests for the hand-written CUDA kernel (`core_cuda_kernel.cluster`),
a third implementation of the same naive O(n*k^2) DP as `core.cluster(...,
vectorize=True)` (C++) and `core_torch.cluster()` (torch tensor ops). Requires
CUDA — every test here is skipped on machines without a CUDA device (this dev
machine included; validated on Bolt/A100 instead). Compares inertia (not exact
labels/centroids) against the SMAWK reference, same reasoning as the other two
backends' test suites — ties can break differently across independent row-
minima search strategies.
"""

import numpy as np
import pytest
import torch

from coreai_opt.deps import _kmeans1d

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_RTOL = 1e-6


def _inertia(points, weights, clusters, centroids):
    points = np.asarray(points, dtype=np.float64)
    weights = np.ones_like(points) if weights is None else np.asarray(weights, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    clusters = np.asarray(clusters)
    return float(np.sum(weights * (points - centroids[clusters]) ** 2))


class TestKmeans1DCudaKernel:
    """Parity tests: `core_cuda_kernel.cluster` vs. the SMAWK reference."""

    @pytest.mark.parametrize("n", [1, 2, 3, 50, 500, 2000])
    @pytest.mark.parametrize("k", [2, 4, 16, 64])
    def test_matches_smawk_unweighted(self, n, k):
        rng = np.random.default_rng(n * 1000 + k)
        array = rng.standard_normal(n)
        kk = min(k, n)

        reference = _kmeans1d.cluster(array, kk)
        result = _kmeans1d.cluster_cuda_kernel(torch.from_numpy(array), kk)

        reference_inertia = _inertia(array, None, reference.clusters, reference.centroids)
        result_inertia = _inertia(
            array, None, result.clusters.cpu().numpy(), result.centroids.cpu().numpy()
        )
        np.testing.assert_allclose(result_inertia, reference_inertia, rtol=_RTOL, atol=1e-9)

    @pytest.mark.parametrize("n", [5, 50, 500])
    @pytest.mark.parametrize("k", [2, 4, 16, 64])
    def test_matches_smawk_weighted(self, n, k):
        rng = np.random.default_rng(n + k)
        array = rng.standard_normal(n)
        weights = rng.choice([1.0, 2.0, 3.0, 4.0], size=n)
        kk = min(k, n)

        reference = _kmeans1d.cluster(array, kk, weights=weights)
        result = _kmeans1d.cluster_cuda_kernel(
            torch.from_numpy(array), kk, weights=torch.from_numpy(weights)
        )

        reference_inertia = _inertia(array, weights, reference.clusters, reference.centroids)
        result_inertia = _inertia(
            array, weights, result.clusters.cpu().numpy(), result.centroids.cpu().numpy()
        )
        np.testing.assert_allclose(result_inertia, reference_inertia, rtol=_RTOL, atol=1e-9)

    def test_centroids_are_ascending_and_labels_in_range(self):
        rng = np.random.default_rng(11)
        array = rng.standard_normal(1000)
        result = _kmeans1d.cluster_cuda_kernel(torch.from_numpy(array), 16)

        centroids = result.centroids
        clusters = result.clusters
        assert torch.all(torch.diff(centroids) >= 0), "centroids must be ascending"
        assert clusters.min() >= 0 and clusters.max() < len(centroids)
        assert clusters.shape == array.shape
        assert centroids.device.type == "cuda"
        assert clusters.device.type == "cuda"

    def test_rejects_non_float64_dtype(self):
        rng = np.random.default_rng(2)
        array = rng.standard_normal(50)
        with pytest.raises(ValueError):
            _kmeans1d.cluster_cuda_kernel(torch.from_numpy(array), 4, dtype=torch.float32)

    def test_weights_equal_repetition(self):
        # Weighting a value by w is equivalent to repeating it w times.
        repeated = torch.tensor([1.0, 1.0, 5.0, 5.0, 5.0, 9.0], dtype=torch.float64)
        unique_values = torch.tensor([1.0, 5.0, 9.0], dtype=torch.float64)
        counts = torch.tensor([2.0, 3.0, 1.0], dtype=torch.float64)

        repeated_result = _kmeans1d.cluster_cuda_kernel(repeated, 2)
        weighted_result = _kmeans1d.cluster_cuda_kernel(unique_values, 2, weights=counts)

        np.testing.assert_allclose(
            np.sort(np.unique(repeated_result.centroids.cpu().numpy())),
            np.sort(weighted_result.centroids.cpu().numpy()),
            rtol=_RTOL,
            atol=1e-9,
        )
