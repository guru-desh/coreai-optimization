# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Correctness tests for the torch-native naive DP (`core_torch.cluster`), an
experimental alternative to `core.cluster(..., vectorize=True)`'s C++ extension —
same algorithm (naive O(n*k^2) DP, no SMAWK), different implementation, kept for a
controlled A/B comparison. Compares inertia (not exact labels/centroids) against
the SMAWK reference, since float32 (the torch default here) and independent
tie-breaking on ties can both cause small, expected divergence from the float64
C++ path — see test_kmeans1d.py's `test_collapse_fewer_than_k_clusters` for the
same reasoning applied to the C++ naive DP vs. SMAWK.
"""

import numpy as np
import pytest
import torch

from coreai_opt.deps import _kmeans1d

_FP32_RTOL = 1e-2
_FP64_RTOL = 1e-6


def _inertia(points, weights, clusters, centroids):
    points = np.asarray(points, dtype=np.float64)
    weights = np.ones_like(points) if weights is None else np.asarray(weights, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    clusters = np.asarray(clusters)
    return float(np.sum(weights * (points - centroids[clusters]) ** 2))


class TestKmeans1DTorch:
    """Parity tests: `core_torch.cluster` vs. the SMAWK reference (`_kmeans1d.cluster`)."""

    @pytest.mark.parametrize(
        "dtype,rtol", [(torch.float32, _FP32_RTOL), (torch.float64, _FP64_RTOL)]
    )
    @pytest.mark.parametrize("n", [1, 2, 3, 50, 500, 2000])
    @pytest.mark.parametrize("k", [2, 4, 16, 64])
    def test_matches_smawk_unweighted(self, dtype, rtol, n, k):
        rng = np.random.default_rng(n * 1000 + k)
        array = rng.standard_normal(n)
        kk = min(k, n)

        reference = _kmeans1d.cluster(array, kk)
        result = _kmeans1d.cluster_torch(torch.from_numpy(array), kk, dtype=dtype, device="cpu")

        reference_inertia = _inertia(array, None, reference.clusters, reference.centroids)
        result_inertia = _inertia(array, None, result.clusters.numpy(), result.centroids.numpy())
        np.testing.assert_allclose(result_inertia, reference_inertia, rtol=rtol, atol=1e-9)

    @pytest.mark.parametrize(
        "dtype,rtol", [(torch.float32, _FP32_RTOL), (torch.float64, _FP64_RTOL)]
    )
    @pytest.mark.parametrize("n", [5, 50, 500])
    @pytest.mark.parametrize("k", [2, 4, 16, 64])
    def test_matches_smawk_weighted(self, dtype, rtol, n, k):
        rng = np.random.default_rng(n + k)
        array = rng.standard_normal(n)
        weights = rng.choice([1.0, 2.0, 3.0, 4.0], size=n)
        kk = min(k, n)

        reference = _kmeans1d.cluster(array, kk, weights=weights)
        result = _kmeans1d.cluster_torch(
            torch.from_numpy(array),
            kk,
            weights=torch.from_numpy(weights),
            dtype=dtype,
            device="cpu",
        )

        reference_inertia = _inertia(array, weights, reference.clusters, reference.centroids)
        result_inertia = _inertia(array, weights, result.clusters.numpy(), result.centroids.numpy())
        np.testing.assert_allclose(result_inertia, reference_inertia, rtol=rtol, atol=1e-9)

    def test_centroids_are_ascending_and_labels_in_range(self):
        rng = np.random.default_rng(11)
        array = rng.standard_normal(1000)
        result = _kmeans1d.cluster_torch(torch.from_numpy(array), 16, device="cpu")

        centroids = result.centroids
        clusters = result.clusters
        assert torch.all(torch.diff(centroids) >= 0), "centroids must be ascending"
        assert clusters.min() >= 0 and clusters.max() < len(centroids)
        assert clusters.shape == array.shape

    def test_defaults_to_cuda_when_available(self):
        rng = np.random.default_rng(3)
        array = rng.standard_normal(100)
        result = _kmeans1d.cluster_torch(torch.from_numpy(array), 4)
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert result.centroids.device.type == expected_device
        assert result.clusters.device.type == expected_device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_matches_smawk_on_cuda(self):
        rng = np.random.default_rng(5)
        array = rng.standard_normal(500)
        reference = _kmeans1d.cluster(array, 16)
        result = _kmeans1d.cluster_torch(torch.from_numpy(array), 16, device="cuda")

        assert result.centroids.device.type == "cuda"
        reference_inertia = _inertia(array, None, reference.clusters, reference.centroids)
        result_inertia = _inertia(
            array, None, result.clusters.cpu().numpy(), result.centroids.cpu().numpy()
        )
        np.testing.assert_allclose(result_inertia, reference_inertia, rtol=_FP32_RTOL, atol=1e-9)
