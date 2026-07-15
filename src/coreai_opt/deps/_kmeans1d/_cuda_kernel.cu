// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-Clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

// Hand-written CUDA kernel implementation of the naive (no-SMAWK) O(n * k^2)
// 1-D k-means DP. Every formula here is a direct translation of already-
// validated code, not new math: the segment-cost formulas mirror
// CostCalculator/WeightedCostCalculator in _core.cpp exactly (unified into a
// single always-weighted path, matching core_torch.py's own
// `weights = ones(n) if weights is None` convention), and the DP
// recurrence/backtrack indexing mirrors core_torch.py's _run_dp/_backtrack
// exactly (0-based "column = point-count in previous clusters" convention,
// not _core.cpp's SMAWK-era column convention, which differs by one).
//
// This exists to fix a specific, measured problem with core_torch.py's
// tensor-ops implementation: its backtrack step calls `.item()` on a CUDA
// tensor once per cluster (k times per weight block), forcing a host-device
// synchronization each time. Real ResNet50/LLM Bolt runs showed the torch
// backend consistently slower than both SMAWK and the C++ naive-DP backend as
// a result. This kernel instead runs the entire DP forward pass AND backtrack
// as a fixed sequence of kernel launches, all queued on the same (default)
// CUDA stream from this file's C++ launcher function -- no host-device sync
// point exists until the caller actually reads the output tensors.
//
// Loaded via torch.utils.cpp_extension.load() and called via ctypes with raw
// device pointers (obtained from Python via tensor.data_ptr()), mirroring
// exactly how core.py's `_dll()` loads and calls the CPU `_core.cpp`
// extension -- this file exports a minimal, mostly-empty Python module (so
// `load()` can import the built .so at all) and does the real work through
// `extern "C"` functions called directly via ctypes, not through Python.
//
// Assumption: neither this code nor its caller uses non-default CUDA streams.
// All kernels here launch on the default stream (no explicit stream argument
// in the <<<>>> launch configuration), and by default so do untouched
// PyTorch tensor ops (e.g. the caller's torch.sort()) -- CUDA guarantees
// in-order execution for kernels queued on the same stream, so the sorted
// input this kernel reads is guaranteed complete before these kernels run,
// with no explicit synchronization required. If a caller ever introduces a
// custom stream anywhere in this call path, this assumption breaks silently.

#include <Python.h>

#include <cfloat>
#include <cstdint>

typedef unsigned long ulong;
typedef int64_t index_t;

#define THREADS_PER_BLOCK 256

// ---------------------------------------------------------------------------
// Device-side cost calculator. Mirrors WeightedCostCalculator::calc(i, j) in
// _core.cpp exactly (i = 0-based segment start, j = 0-based segment end,
// inclusive). Always weighted -- the caller passes an all-ones weights array
// for the unweighted case, so there is only one code path here to get right.
// ---------------------------------------------------------------------------

__device__ inline double weighted_segment_cost(
        const double* cumw, const double* cumsum, const double* cumsum2,
        ulong i, ulong j) {
    if (j < i) return 0.0;
    double w = cumw[j + 1] - cumw[i];
    double mu = (cumsum[j + 1] - cumsum[i]) / w;
    double result = cumsum2[j + 1] - cumsum2[i];
    result += w * (mu * mu);
    result -= (2.0 * mu) * (cumsum[j + 1] - cumsum[i]);
    return result;
}

// ---------------------------------------------------------------------------
// Prefix sums: single thread, O(n). Not the bottleneck this kernel targets --
// see core.py's docstring on the O(n^2)-time regime this whole naive DP is
// viable for at all (n up to a few thousand); a parallel scan would add real
// implementation risk for no measured benefit at that scale.
// ---------------------------------------------------------------------------

__global__ void prefix_sums_kernel(
        const double* sorted_values, const double* sorted_weights,
        double* cumw, double* cumsum, double* cumsum2, ulong n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    cumw[0] = 0.0;
    cumsum[0] = 0.0;
    cumsum2[0] = 0.0;
    for (ulong i = 0; i < n; ++i) {
        double x = sorted_values[i];
        double w = sorted_weights[i];
        cumw[i + 1] = cumw[i] + w;
        cumsum[i + 1] = cumsum[i] + w * x;
        cumsum2[i + 1] = cumsum2[i] + w * x * x;
    }
}

// Parallel fill of the first `count` elements of `buf` with `value` (grid-
// stride so a single launch is safe for any `count`, though callers here only
// ever use it for the full-row init or a single element).
__global__ void fill_kernel(double* buf, ulong count, double value) {
    for (ulong idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += blockDim.x * gridDim.x) {
        buf[idx] = value;
    }
}

// ---------------------------------------------------------------------------
// DP layer kernel: one block per row `i` (0-based, representing the prefix of
// the first `i + 1` sorted points). Threads in the block cooperatively scan
// column `j` = 0..i (0-based count of points assigned to previous clusters --
// core_torch.py's convention, NOT _core.cpp's SMAWK-column convention, which
// differs by one) -- the naive, no-SMAWK row-minima search, done in parallel
// across threads instead of a sequential loop -- and reduce to the row
// minimum via a standard shared-memory tree reduction.
//
// `prev[j]` must hold D[clusters - 1][j] (cost of the first j points using
// `clusters - 1` clusters) for j = 0..n-1, with prev[0] = 0 only when
// clusters - 1 == 0 (the base case: 0 clusters over 0 points), else +inf --
// this is the caller's responsibility to set up per layer (see
// cluster_naive_dp_cuda below), mirroring core_torch.py's `cost_prev` exactly.
//
// Writes `cur[i + 1] = D[clusters][i + 1]` directly -- not `cur[i]` -- so the
// output buffer is immediately in the correct shape to serve as the NEXT
// layer's `prev` with no separate shift step (core_torch.py's tensor version
// needs an explicit `torch.cat([inf_tail, cost_cur[:-1]])` shift for this same
// reason; writing directly to `i + 1` here bakes the shift into the write
// location instead).
// ---------------------------------------------------------------------------

__global__ void dp_layer_kernel(
        const double* cumw, const double* cumsum, const double* cumsum2,
        const double* prev, double* cur, index_t* split_index, ulong n) {
    ulong i = blockIdx.x;
    if (i >= n) return;

    __shared__ double best_cost[THREADS_PER_BLOCK];
    __shared__ index_t best_j[THREADS_PER_BLOCK];

    double local_best_cost = DBL_MAX;
    index_t local_best_j = 0;
    for (ulong j = threadIdx.x; j <= i; j += blockDim.x) {
        double seg = weighted_segment_cost(cumw, cumsum, cumsum2, j, i);
        double candidate = prev[j] + seg;
        if (candidate < local_best_cost) {
            local_best_cost = candidate;
            local_best_j = (index_t)j;
        }
    }
    best_cost[threadIdx.x] = local_best_cost;
    best_j[threadIdx.x] = local_best_j;
    __syncthreads();

    for (ulong stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride &&
            best_cost[threadIdx.x + stride] < best_cost[threadIdx.x]) {
            best_cost[threadIdx.x] = best_cost[threadIdx.x + stride];
            best_j[threadIdx.x] = best_j[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        cur[i + 1] = best_cost[0];
        split_index[i] = best_j[0];
    }
}

// ---------------------------------------------------------------------------
// Backtrack: single thread, sequential walk from `clusters = k` downto `1`
// (inherently sequential -- each step's segment start depends on the
// previous step's split), entirely on-device. Mirrors core_torch.py's
// `_backtrack` exactly: `end` is a 1-based count of points not yet assigned;
// `split_index[(layer - 1) * n + (end - 1)]` gives the 0-based start index
// `j` of the current segment; the segment covers 0-based indices `j..end-1`.
// ---------------------------------------------------------------------------

__global__ void backtrack_kernel(
        const double* cumw, const double* cumsum, const index_t* split_index,
        double* centroids, index_t* sorted_labels, ulong n, ulong k) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    ulong end = n;
    for (ulong layer = k; layer >= 1; --layer) {
        ulong j = (ulong)split_index[(layer - 1) * n + (end - 1)];
        double weight_total = cumw[end] - cumw[j];
        centroids[layer - 1] = (cumsum[end] - cumsum[j]) / weight_total;
        for (ulong idx = j; idx < end; ++idx) {
            sorted_labels[idx] = (index_t)(layer - 1);
        }
        end = j;
        if (layer == 1) break;  // avoid unsigned wraparound on the loop decrement
    }
}

// ---------------------------------------------------------------------------
// Host-side launcher. All arguments are device pointers (obtained by the
// Python caller via tensor.data_ptr() and passed through ctypes as plain
// integers) except `n`/`k`. Every kernel below launches on the default
// stream with no intervening host synchronization -- the function returns as
// soon as the last kernel is enqueued, not when the work completes.
// ---------------------------------------------------------------------------

extern "C" {

#if defined(_WIN32) || defined(__CYGWIN__)
__declspec(dllexport)
#endif
void cluster_naive_dp_cuda(
        double* sorted_values,   // (n,)
        double* sorted_weights,  // (n,) -- caller passes all-ones if unweighted
        double* cumw,            // scratch, (n + 1,)
        double* cumsum,          // scratch, (n + 1,)
        double* cumsum2,         // scratch, (n + 1,)
        double* buf_a,           // scratch, (n + 1,) -- ping-pong DP row buffer
        double* buf_b,           // scratch, (n + 1,) -- ping-pong DP row buffer
        index_t* split_index,    // output, (k, n)
        double* centroids,       // output, (k,)
        index_t* sorted_labels,  // output, (n,)
        ulong n,
        ulong k) {
    prefix_sums_kernel<<<1, 1>>>(sorted_values, sorted_weights, cumw, cumsum, cumsum2, n);

    double* prev = buf_a;
    double* cur = buf_b;

    // Base case D[0][j]: 0 clusters over j points. j = 0 -> cost 0 (valid,
    // vacuously); j > 0 -> infeasible (+inf). prev[0..n-1] are the only
    // indices dp_layer_kernel ever reads as a "j" column.
    ulong fill_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    fill_kernel<<<fill_blocks, THREADS_PER_BLOCK>>>(prev, n, DBL_MAX);
    fill_kernel<<<1, 1>>>(prev, 1, 0.0);

    for (ulong layer = 1; layer <= k; ++layer) {
        // D[layer][0] is never written by dp_layer_kernel (rows start at
        // i = 0, writing cur[1..n]); set the point-count-0 base case here,
        // matching prev's initialization above (always +inf for layer >= 1).
        fill_kernel<<<1, 1>>>(cur, 1, DBL_MAX);
        dp_layer_kernel<<<n, THREADS_PER_BLOCK>>>(
            cumw, cumsum, cumsum2, prev, cur, split_index + (layer - 1) * n, n);
        double* tmp = prev;
        prev = cur;
        cur = tmp;
    }

    backtrack_kernel<<<1, 1>>>(cumw, cumsum, split_index, centroids, sorted_labels, n, k);
}

} // extern "C"

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _cuda_kernel_module = {
    PyModuleDef_HEAD_INIT,
    "kmeans1d._cuda_kernel",
    NULL,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__cuda_kernel(void) {
    return PyModule_Create(&_cuda_kernel_module);
}
