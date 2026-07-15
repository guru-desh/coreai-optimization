# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Fixed-value tests for the vendored 1D k-means, backed by hardcoded
literals for small inputs and checked-in golden ``.npz`` files (see
``assets/kmeans1d/``) for larger ones. Our extension JIT-compiles
``_core.cpp`` on first use and caches the artifact for the session, so the
compile cost is paid once. Most tests are parametrized over ``vectorize``
(SMAWK vs. the naive O(n * k^2) DP): both compute the identical global
optimum by construction (same D/T table; SMAWK is only a faster way to find
the same row-minima), so the same golden values apply to both. The
``vectorize=True`` path is not parametrized into
``test_matches_known_clustering_at_scale`` (n up to 100,000) since an O(n^2)
DP at that scale is infeasible (~10**10 cells).
"""

from pathlib import Path

import numpy as np
import pytest

from coreai_opt.deps import _kmeans1d

_RTOL = 1e-9
_ATOL = 1e-9

_ASSETS_DIR = Path(__file__).parent / "assets" / "kmeans1d"


def _inertia(points, weights, clusters, centroids):
    """Weighted within-cluster sum of squared deviations."""
    points = np.asarray(points, dtype=np.float64)
    weights = np.ones_like(points) if weights is None else np.asarray(weights, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    clusters = np.asarray(clusters)
    return float(np.sum(weights * (points - centroids[clusters]) ** 2))


@pytest.mark.parametrize("vectorize", [False, True])
class TestKmeans1D:
    """Fixed-value and behavior tests for the vendored cluster() surface."""

    # -- Small, distinct-valued inputs: hardcode expected values directly --

    def test_cluster(self, vectorize):
        array = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
        result = _kmeans1d.cluster(array, 4, vectorize=vectorize)
        np.testing.assert_array_equal(result.clusters, [1, 1, 1, 0, 3, 3, 3, 2, 2, 2])
        np.testing.assert_allclose(
            result.centroids, [-50.0, 4.1, 94.0, 200.5], rtol=_RTOL, atol=_ATOL
        )

    def test_cluster_with_weights(self, vectorize):
        array = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
        weights = [1, 1, 1, 0.125, 4, 1, 1, 3, 2, 2]
        result = _kmeans1d.cluster(array, 4, weights=weights, vectorize=vectorize)
        np.testing.assert_array_equal(result.clusters, [0, 0, 0, 0, 3, 3, 3, 1, 2, 2])
        np.testing.assert_allclose(
            result.centroids, [1.936, 80.0, 101.0, 200.35], rtol=_RTOL, atol=_ATOL
        )

    def test_weights_vs_repetition(self, vectorize):
        values = [10, 24, 16, 12, 20]
        weights = [3, 1, 4, 2, 3]
        weighted = _kmeans1d.cluster(values, 2, weights=weights, vectorize=vectorize)
        np.testing.assert_array_equal(weighted.clusters, [0, 1, 1, 0, 1])
        np.testing.assert_allclose(weighted.centroids, [10.8, 18.5], rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(
            _inertia(values, weights, weighted.clusters, weighted.centroids),
            66.8,
            rtol=_RTOL,
            atol=_ATOL,
        )

        # Weighting by integer counts equals repeating each value that many times.
        repeated = np.repeat(values, weights)
        repeated_result = _kmeans1d.cluster(repeated, 2, vectorize=vectorize)
        np.testing.assert_allclose(
            np.sort(repeated_result.centroids),
            np.sort(weighted.centroids),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            _inertia(repeated, None, repeated_result.clusters, repeated_result.centroids),
            66.8,
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_k_greater_than_n_is_clamped(self, vectorize):
        # 5 distinct values, k=16 clamps to 5: each value is trivially its own cluster.
        array = np.array([1.0, 5.0, 9.0, 13.0, 17.0])
        result = _kmeans1d.cluster(array, 16, vectorize=vectorize)
        np.testing.assert_array_equal(result.clusters, [0, 1, 2, 3, 4])
        np.testing.assert_allclose(result.centroids, array, rtol=_RTOL, atol=_ATOL)

    @pytest.mark.parametrize(
        "array, k, expected_centroids, expected_reconstruction",
        [
            ([5.0, 5.0, 5.0], 3, [0.0, 0.0, 5.0], [5.0, 5.0, 5.0]),
            ([1.0, 1.0, 9.0], 3, [0.0, 1.0, 9.0], [1.0, 1.0, 9.0]),
            ([-5.0, -5.0, -5.0], 3, [0.0, 0.0, -5.0], [-5.0, -5.0, -5.0]),
            ([2.0, 2.0, 2.0, 2.0], 4, [0.0, 0.0, 0.0, 2.0], [2.0, 2.0, 2.0, 2.0]),
        ],
        ids=["all-equal", "two-distinct", "all-negative-equal", "four-equal"],
    )
    def test_collapse_fewer_than_k_clusters(
        self, vectorize, array, k, expected_centroids, expected_reconstruction
    ):
        # When k exceeds the number of distinct values, the optimum uses fewer than
        # k clusters; unused leading centroids are zero-filled (including the
        # non-ascending padding in "all-negative-equal" below) — but only for SMAWK's
        # specific tie-breaking. These inputs are fully degenerate (every partition
        # costs exactly the same), so the naive DP's independent tie-breaking can
        # legitimately pick a different, equally optimal partition (e.g. one point
        # per cluster instead of SMAWK's "concentrate in the last cluster"), which
        # changes the raw centroids array without being a correctness bug. Only the
        # reconstructed value each point maps to is guaranteed stable across either
        # backend; the exact zero-filled centroids array is only asserted for SMAWK.
        result = _kmeans1d.cluster(np.array(array), k, vectorize=vectorize)
        centroids = np.asarray(result.centroids)
        clusters = np.asarray(result.clusters)
        if not vectorize:
            np.testing.assert_allclose(centroids, expected_centroids, rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(
            centroids[clusters], expected_reconstruction, rtol=_RTOL, atol=_ATOL
        )

    def test_explicit_duplicates_reconstruction_matches(self, vectorize):
        array = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 9.0, 9.0, 9.0, 9.0])
        result = _kmeans1d.cluster(array, 3, vectorize=vectorize)
        centroids = np.asarray(result.centroids)
        clusters = np.asarray(result.clusters)
        np.testing.assert_allclose(centroids, [1.0, 5.0, 9.0], rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(centroids[clusters], array, rtol=_RTOL, atol=_ATOL)

    # -- Larger/random inputs: compare against golden values in assets/kmeans1d/ --

    @pytest.mark.parametrize("n", [2, 5, 50, 2_000])
    @pytest.mark.parametrize("k", [2, 4, 16, 64, 256])
    def test_matches_known_unweighted_clustering(self, vectorize, n, k):
        golden = np.load(_ASSETS_DIR / "unweighted.npz")
        result = _kmeans1d.cluster(golden[f"array_n{n}_k{k}"], k, vectorize=vectorize)

        np.testing.assert_array_equal(result.clusters, golden[f"clusters_n{n}_k{k}"])
        np.testing.assert_allclose(
            result.centroids, golden[f"centroids_n{n}_k{k}"], rtol=_RTOL, atol=_ATOL
        )

    @pytest.mark.parametrize("n", [5, 500, 2_000])
    @pytest.mark.parametrize("k", [2, 4, 16, 64, 256])
    def test_matches_known_weighted_clustering(self, vectorize, n, k):
        golden = np.load(_ASSETS_DIR / "weighted.npz")
        result = _kmeans1d.cluster(
            golden[f"array_n{n}_k{k}"],
            k,
            weights=golden[f"weights_n{n}_k{k}"],
            vectorize=vectorize,
        )

        np.testing.assert_array_equal(result.clusters, golden[f"clusters_n{n}_k{k}"])
        np.testing.assert_allclose(
            result.centroids, golden[f"centroids_n{n}_k{k}"], rtol=_RTOL, atol=_ATOL
        )

    def test_matches_known_fp16_clustering(self, vectorize):
        # fp16 has few representable values, so duplicates (ties) are expected; only
        # the reconstruction is compared, per test_collapse_fewer_than_k_clusters.
        golden = np.load(_ASSETS_DIR / "fp16_duplicates.npz")
        result = _kmeans1d.cluster(golden["array"], 16, vectorize=vectorize)
        centroids = np.asarray(result.centroids)
        clusters = np.asarray(result.clusters)

        np.testing.assert_allclose(centroids, golden["centroids"], rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(
            centroids[clusters], golden["reconstruction"], rtol=_RTOL, atol=_ATOL
        )

    def test_matches_known_weighted_integer_counts_clustering(self, vectorize):
        # Mirrors the fast/dedup path: cluster unique values weighted by counts.
        golden = np.load(_ASSETS_DIR / "weighted_integer_counts.npz")
        result = _kmeans1d.cluster(
            golden["values"], 16, weights=golden["counts"], vectorize=vectorize
        )

        np.testing.assert_array_equal(result.clusters, golden["clusters"])
        np.testing.assert_allclose(result.centroids, golden["centroids"], rtol=_RTOL, atol=_ATOL)

    def test_vectorize_matches_smawk_at_moderate_n(self, vectorize):
        # Direct backend-parity check independent of the golden .npz files: at a
        # size and k range both backends handle comfortably, SMAWK (vectorize=False)
        # and the naive DP (vectorize=True) must reach the same optimum. Compares
        # inertia rather than exact labels/centroids since continuous random data
        # can, in principle, produce DP ties that the two row-minima search
        # strategies break differently (see test_collapse_fewer_than_k_clusters).
        rng = np.random.default_rng(42)
        array = rng.standard_normal(2_000)
        reference = _kmeans1d.cluster(array, 64, vectorize=False)
        result = _kmeans1d.cluster(array, 64, vectorize=vectorize)
        reference_inertia = _inertia(array, None, reference.clusters, reference.centroids)
        result_inertia = _inertia(array, None, result.clusters, result.centroids)
        np.testing.assert_allclose(result_inertia, reference_inertia, rtol=_RTOL, atol=_ATOL)

    # -- Property-based behavior (no oracle, no golden file needed) --

    def test_weights_equal_repetition(self, vectorize):
        # Weighting a value by w is equivalent to repeating it w times.
        repeated = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 9.0])
        unique_values = np.array([1.0, 5.0, 9.0])
        counts = np.array([2.0, 3.0, 1.0])

        repeated_result = _kmeans1d.cluster(repeated, 2, vectorize=vectorize)
        weighted_result = _kmeans1d.cluster(unique_values, 2, weights=counts, vectorize=vectorize)

        np.testing.assert_allclose(
            np.sort(np.unique(repeated_result.centroids)),
            np.sort(weighted_result.centroids),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_centroids_are_ascending_and_labels_in_range(self, vectorize):
        rng = np.random.default_rng(11)
        array = rng.standard_normal(1000)
        result = _kmeans1d.cluster(array, 16, vectorize=vectorize)

        centroids = np.asarray(result.centroids)
        clusters = np.asarray(result.clusters)
        assert np.all(np.diff(centroids) >= 0), "centroids must be ascending"
        assert clusters.min() >= 0 and clusters.max() < len(centroids)
        assert clusters.shape == array.shape
        assert centroids.dtype == np.float64

    def test_accepts_list_and_preserves_original_order(self, vectorize):
        # A plain Python list (the doc-script call style) must work, and labels are
        # returned in the original (unsorted) order.
        result = _kmeans1d.cluster([9.0, 0.1, 9.1, 0.0], 2, vectorize=vectorize)
        clusters = np.asarray(result.clusters)
        centroids = np.asarray(result.centroids)
        # Points 0,2 are the large cluster; 1,3 the small one.
        assert clusters[0] == clusters[2]
        assert clusters[1] == clusters[3]
        assert clusters[0] != clusters[1]
        np.testing.assert_allclose(centroids, [0.05, 9.05], rtol=_RTOL, atol=_ATOL)


class TestKmeans1DAtScale:
    """SMAWK-only tests at scales infeasible for the O(n^2) vectorize DP."""

    @pytest.mark.parametrize("n", [10_000, 20_000, 50_000, 100_000])
    @pytest.mark.parametrize("k", [4, 64])
    def test_matches_known_clustering_at_scale(self, n, k):
        # O(k*n) time and memory: n=1e5 x k=64 is ~100 MB per DP table. 1e6/1e7 are
        # infeasible for the exact DP. vectorize=True is not exercised here: an
        # O(n^2) DP at this scale (~10**10 cells) is infeasible regardless of
        # backend.
        golden = np.load(_ASSETS_DIR / "large_n.npz")
        result = _kmeans1d.cluster(golden[f"array_n{n}_k{k}"], k)

        np.testing.assert_array_equal(result.clusters, golden[f"clusters_n{n}_k{k}"])
        np.testing.assert_allclose(
            result.centroids, golden[f"centroids_n{n}_k{k}"], rtol=_RTOL, atol=_ATOL
        )
