# Copyright © 2026 Apple Inc.

from collections.abc import Sequence

from coremltools._deps.kmeans1d import Clustered, cluster as _coremltools_cluster


def cluster(array: Sequence[float], k: int, *, weights: Sequence[float] | None = None) -> Clustered:
    """
    :param array: A sequence of floats
    :param k: Number of clusters (int)
    :param weights: Sequence of weights (if provided, must have same length as `array`)
    :return: A tuple with (clusters, centroids)
    """
    return _coremltools_cluster(array, k, weights=weights)
