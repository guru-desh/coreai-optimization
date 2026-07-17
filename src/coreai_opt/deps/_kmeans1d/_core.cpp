// MIT License
//
// Copyright (c) 2019 Daniel Steinberg
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Copyright © 2026 Apple Inc.

#include <Python.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

using namespace std;

typedef unsigned long ulong;

/*
 *  Internal implementation of the SMAWK algorithm.
 */
template <typename T>
void _smawk(
        const vector<ulong>& rows,
        const vector<ulong>& cols,
        const function<T(ulong, ulong)>& lookup,
        vector<ulong>* result) {
    // Recursion base case
    if (rows.size() == 0) return;

    // ********************************
    // * REDUCE
    // ********************************

    vector<ulong> _cols;  // Stack of surviving columns
    for (ulong col : cols) {
        while (true) {
            if (_cols.size() == 0) break;
            ulong row = rows[_cols.size() - 1];
            if (lookup(row, col) >= lookup(row, _cols.back()))
                break;
            _cols.pop_back();
        }
        if (_cols.size() < rows.size())
            _cols.push_back(col);
    }

    // Call recursively on odd-indexed rows
    vector<ulong> odd_rows;
    for (ulong i = 1; i < rows.size(); i += 2) {
        odd_rows.push_back(rows[i]);
    }
    _smawk(odd_rows, _cols, lookup, result);

    unordered_map<ulong, ulong> col_idx_lookup;
    for (ulong idx = 0; idx < _cols.size(); ++idx) {
        col_idx_lookup[_cols[idx]] = idx;
    }

    // ********************************
    // * INTERPOLATE
    // ********************************

    // Fill-in even-indexed rows
    ulong start = 0;
    for (ulong r = 0; r < rows.size(); r += 2) {
        ulong row = rows[r];
        ulong stop = _cols.size() - 1;
        if (r < rows.size() - 1)
            stop = col_idx_lookup[(*result)[rows[r + 1]]];
        ulong argmin = _cols[start];
        T min = lookup(row, argmin);
        for (ulong c = start + 1; c <= stop; ++c) {
            T value = lookup(row, _cols[c]);
            if (c == start || value < min) {
                argmin = _cols[c];
                min = value;
            }
        }
        (*result)[row] = argmin;
        start = stop;
    }
}

/*
 *  Interface for the SMAWK algorithm, for finding the minimum value in each row
 *  of an implicitly-defined totally monotone matrix.
 */
template <typename T>
vector<ulong> smawk(
        const ulong num_rows,
        const ulong num_cols,
        const function<T(ulong, ulong)>& lookup) {
    vector<ulong> result;
    result.resize(num_rows);
    vector<ulong> rows(num_rows);
    iota(begin(rows), end(rows), 0);
    vector<ulong> cols(num_cols);
    iota(begin(cols), end(cols), 0);
    _smawk<T>(rows, cols, lookup, &result);
    return result;
}

/*
 *  Calculates cluster costs in O(1) using prefix sum arrays.
 */
class CostCalculator {
    vector<double> cumsum;
    vector<double> cumsum2;

  public:
    CostCalculator(const vector<double>& vec, ulong n, const vector<ulong>& /*sort_idxs*/) {
        cumsum.push_back(0.0);
        cumsum2.push_back(0.0);
        for (ulong i = 0; i < n; ++i) {
            double x = vec[i];
            cumsum.push_back(x + cumsum[i]);
            cumsum2.push_back(x * x + cumsum2[i]);
        }
    }

    double weight(ulong i, ulong j) {
        return (i <= j) ? 1 + j - i : 0;
    }

    double calc(ulong i, ulong j) {
        if (j < i) return 0.0;
        double mu = (cumsum[j + 1] - cumsum[i]) / (j - i + 1);
        double result = cumsum2[j + 1] - cumsum2[i];
        result += (j - i + 1) * (mu * mu);
        result -= (2 * mu) * (cumsum[j + 1] - cumsum[i]);
        return result;
    }
};

/*
 *  Weighted version of the CostCalculator
 */
class WeightedCostCalculator {
    vector<double> cumw;
    vector<double> cumsum;
    vector<double> cumsum2;

  public:
    WeightedCostCalculator(
            const vector<double>& vec,
            ulong n,
            const vector<ulong>& sort_idxs,
            const double* unsorted_weights) {
        vector<double> sorted_weights(n);
        for (ulong i = 0; i < n; ++i) {
            sorted_weights[i] = unsorted_weights[sort_idxs[i]];
        }
        cumw.push_back(0.0);
        cumsum.push_back(0.0);
        cumsum2.push_back(0.0);
        for (ulong i = 0; i < n; ++i) {
            double x = vec[i];
            double w = sorted_weights[i];
            cumw.push_back(w + cumw[i]);
            cumsum.push_back(w * x + cumsum[i]);
            cumsum2.push_back(w * x * x + cumsum2[i]);
        }
    }

    double weight(ulong i, ulong j) {
        return (i <= j) ? cumw[j + 1] - cumw[i] : 0.0;
    }

    double calc(ulong i, ulong j) {
        if (j < i) return 0.0;
        double w = weight(i, j);
        double mu = (cumsum[j + 1] - cumsum[i]) / w;
        double result = cumsum2[j + 1] - cumsum2[i];
        result += w * (mu * mu);
        result -= (2 * mu) * (cumsum[j + 1] - cumsum[i]);
        return result;
    }
};

template <typename T>
class Matrix {
    vector<T> data;
    ulong num_rows;
    ulong num_cols;

  public:
    Matrix(ulong num_rows, ulong num_cols) {
        this->num_rows = num_rows;
        this->num_cols = num_cols;
        data.resize(num_rows * num_cols);
    }

    inline T get(ulong i, ulong j) {
        return data[i * num_cols + j];
    }

    inline void set(ulong i, ulong j, T value) {
        data[i * num_cols + j] = value;
    }
};

// Computes the full row of DP values for `count` clusters over `len`
// (local) positions, given an arbitrary cost lookup `calc(a,b)` for local
// indices a<=b in [0,len): result[i] = optimal cost of partitioning
// local [0,i] into exactly `count` clusters under that cost function.
// Identical recurrence and SMAWK usage (including the col=i<j-1?i:j-1
// clamp for out-of-range candidates) to cluster_impl's original
// single-pass loop -- only 2 rows of D are live at once (same technique
// as the D-row-reduction change), so this uses O(len) space.
template <typename CalcFn>
vector<double> generic_dp_row(ulong len, ulong count, CalcFn&& calc) {
    Matrix<double> D(2, len);
    for (ulong i = 0; i < len; ++i) {
        D.set(0, i, calc(0, i));
    }
    for (ulong k_ = 1; k_ < count; ++k_) {
        ulong prev_row = (k_ - 1) % 2;
        ulong curr_row = k_ % 2;
        auto C = [&D, &prev_row, &calc](ulong i, ulong j) -> double {
            ulong col = i < j - 1 ? i : j - 1;
            return D.get(prev_row, col) + calc(j, i);
        };
        vector<ulong> row_argmins = smawk<double>(len, len, C);
        for (ulong i = 0; i < row_argmins.size(); ++i) {
            D.set(curr_row, i, C(i, row_argmins[i]));
        }
    }
    ulong final_row = (count - 1) % 2;
    vector<double> result(len);
    for (ulong i = 0; i < len; ++i) result[i] = D.get(final_row, i);
    return result;
}

// result[i] = optimal cost of partitioning [lo, lo+i] into `count` clusters.
template <typename CostCalculatorType>
vector<double> forward_dp_row(
        CostCalculatorType& cost_calculator, ulong lo, ulong hi, ulong count) {
    ulong len = hi - lo;
    return generic_dp_row(len, count, [&cost_calculator, lo](ulong a, ulong b) {
        return cost_calculator.calc(lo + a, lo + b);
    });
}

// result[i] = optimal cost of partitioning [hi-1-i, hi-1] (the last i+1
// elements of [lo, hi)) into `count` clusters. Computed by handing
// generic_dp_row a cost lookup mirrored within [lo, hi) -- local index a
// maps to global position hi-1-a, reversing order -- rather than deriving
// a separate clamp for a "backward" recurrence: reusing forward's exact,
// already-correct SMAWK/clamp logic under a reflected cost function avoids
// re-deriving those invariants (and getting them subtly wrong) from scratch.
template <typename CostCalculatorType>
vector<double> backward_dp_row(
        CostCalculatorType& cost_calculator, ulong lo, ulong hi, ulong count) {
    ulong len = hi - lo;
    return generic_dp_row(len, count, [&cost_calculator, hi](ulong a, ulong b) {
        return cost_calculator.calc(hi - 1 - b, hi - 1 - a);
    });
}

// Hirschberg-style divide-and-conquer reconstruction of the optimal
// clustering, avoiding the O(kn) T matrix: recursively split the cluster
// budget `count` in half, run a forward DP for the left half and a
// backward DP for the right half (each O(hi-lo) space via the 2-row
// technique), find where they meet at the true optimum, and recurse on
// each side. O(kn) total time (a constant factor over the direct DP, per
// section 3 of Gronlund et al., 2017), O(n) space overall since only one
// "level" of forward/backward rows is live on the call stack at a time.
template <typename CostCalculatorType>
void hirschberg_solve(
        CostCalculatorType& cost_calculator,
        const vector<double>& sorted_array,
        ulong lo, ulong hi, ulong count, ulong cluster_label_offset,
        vector<double>& sorted_clusters, double* centroids) {
    if (count == 1) {
        // Base case: the whole [lo, hi) range is one cluster. Same
        // incremental weighted-mean formula as the original backtracking
        // loop, applied to this one segment.
        double centroid = 0.0;
        for (ulong i = lo; i < hi; ++i) {
            sorted_clusters[i] = cluster_label_offset;
            centroid += (
                (sorted_array[i] - centroid)
                * cost_calculator.weight(i, i)
                / cost_calculator.weight(lo, i)
            );
        }
        centroids[cluster_label_offset] = centroid;
        return;
    }

    ulong mid_k = count / 2;
    vector<double> fwd = forward_dp_row(cost_calculator, lo, hi, mid_k);
    vector<double> bwd = backward_dp_row(cost_calculator, lo, hi, count - mid_k);

    // Find split point m in [lo, hi) -- the last element of the left
    // part -- minimizing fwd[m-lo] + bwd[hi-2-m], leaving enough elements
    // on each side for their respective cluster counts.
    ulong best_m = lo + mid_k - 1;
    double best_cost = fwd[best_m - lo] + bwd[hi - 2 - best_m];
    for (ulong m = lo + mid_k; m <= hi - 1 - (count - mid_k); ++m) {
        double cost = fwd[m - lo] + bwd[hi - 2 - m];
        if (cost < best_cost) {
            best_cost = cost;
            best_m = m;
        }
    }

    hirschberg_solve(
        cost_calculator, sorted_array, lo, best_m + 1, mid_k,
        cluster_label_offset, sorted_clusters, centroids
    );
    hirschberg_solve(
        cost_calculator, sorted_array, best_m + 1, hi, count - mid_k,
        cluster_label_offset + mid_k, sorted_clusters, centroids
    );
}

template <typename CostCalculatorType, typename... CostArgsTypes>
void cluster_impl(
        const double* array,
        ulong n,
        ulong k,
        ulong* clusters,
        double* centroids,
        CostArgsTypes... args) {
    // ***************************************************
    // * Sort input array and save info for de-sorting
    // ***************************************************

    vector<ulong> sort_idxs(n);
    iota(sort_idxs.begin(), sort_idxs.end(), 0);
    sort(
        sort_idxs.begin(),
        sort_idxs.end(),
        [&array](ulong a, ulong b) {return array[a] < array[b];});
    vector<ulong> undo_sort_lookup(n);
    vector<double> sorted_array(n);
    for (ulong i = 0; i < n; ++i) {
        sorted_array[i] = array[sort_idxs[i]];
        undo_sort_lookup[sort_idxs[i]] = i;
    }

    // ***************************************************
    // * Set D and extract cluster assignments via Hirschberg's technique
    // ***************************************************

    // Algorithm as presented in section 2.2 of (Gronlund et al., 2017),
    // with the backtracking step's O(kn) T matrix replaced by the
    // divide-and-conquer reconstruction in hirschberg_solve() (section 3).

    CostCalculatorType cost_calculator(sorted_array, n, sort_idxs, args...);
    vector<double> sorted_clusters(n);
    hirschberg_solve(
        cost_calculator, sorted_array, 0, n, k, 0, sorted_clusters, centroids
    );

    // ***************************************************
    // * Order cluster assignments to match de-sorted
    // * ordering
    // ***************************************************

    for (ulong i = 0; i < n; ++i) {
        clusters[i] = sorted_clusters[undo_sort_lookup[i]];
    }
}

extern "C" {
// "__declspec(dllexport)" causes the function to be exported when compiling on Windows.
// Otherwise, the function is not exported and the code raises
//   "AttributeError: function 'cluster' not found".
// Exporting is a Windows platform requirement, not just a Visual Studio requirement
// (https://stackoverflow.com/a/22288874/1509433). The _WIN32 macro covers the Visual
// Studio compiler (MSVC) and MinGW. The __CYGWIN__ macro covers gcc and clang under
// Cygwin.
#if defined(_WIN32) || defined(__CYGWIN__)
__declspec(dllexport)
#endif
void cluster(
        double* array,
        ulong n,
        ulong k,
        ulong* clusters,
        double* centroids) {
    cluster_impl<CostCalculator>(array, n, k, clusters, centroids);
}

#if defined(_WIN32) || defined(__CYGWIN__)
__declspec(dllexport)
#endif
void cluster_with_weights(
        double* array,
        double* weights,
        ulong n,
        ulong k,
        ulong* clusters,
        double* centroids) {
    cluster_impl<WeightedCostCalculator, const double*>(
        array, n, k, clusters, centroids, weights
    );
}
} // extern "C"

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef _coremodule = {
    PyModuleDef_HEAD_INIT,
    "kmeans1d._core",
    NULL,
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&_coremodule);
}
