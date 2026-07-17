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
#include <numeric>
#include <vector>

using namespace std;

typedef unsigned long ulong;

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

// Fills row k_ of the DP (D_out/T_out, size n) from row k_-1 (D_prev), using
// Wilber's O(n log n) stack-based technique for the row-minimization
// subproblem SMAWK also solves, as an alternative to SMAWK's matrix-search
// recursion. Reformulating cluster_impl's per-cell recurrence
//   C(i,j) = D_prev[i<j-1 ? i : j-1] + calc(j,i)
// by cases on j shows only 2 shapes actually matter: j in [1,i] is a real
// split (last cluster = [j,i], cost D_prev[j-1] + calc(j,i)); every j
// outside that range collapses to the constant D_prev[i] (an unused/empty
// last cluster -- how the DP naturally represents "k_ clusters aren't all
// needed yet"). So D_out[i] = min(D_prev[i], min_{j=1}^{i} (D_prev[j-1] +
// calc(j,i))), and the inner min is the classical concave-1D-DP-speedup
// shape: candidate j becomes available exactly at row i=j, and (by the
// quadrangle inequality calc() satisfies) each candidate's domain of
// optimality is a contiguous range of rows, discovered here via a
// monotonic stack of candidates plus binary search for each new
// candidate's crossover point against the current stack top.
template <typename CostCalculatorType>
void wilber_fill_row(
        ulong n,
        const vector<double>& D_prev,
        CostCalculatorType& cost_calculator,
        vector<double>* D_out,
        vector<ulong>* T_out) {
    vector<ulong> cand;   // candidate split points j, increasing
    vector<ulong> start;  // start[t] = first row where cand[t] is optimal
    ulong front = 0;

    auto value_at = [&](ulong j, ulong row) -> double {
        return D_prev[j - 1] + cost_calculator.calc(j, row);
    };

    for (ulong i = 0; i < n; ++i) {
        ulong new_j = i + 1;
        while (!cand.empty()) {
            ulong top_j = cand.back();
            ulong top_start = start.back();
            if (value_at(new_j, top_start) <= value_at(top_j, top_start)) {
                cand.pop_back();
                start.pop_back();
                if (front > cand.size()) front = cand.empty() ? 0 : cand.size() - 1;
            } else {
                break;
            }
        }
        ulong lo = max(new_j, cand.empty() ? new_j : start.back());
        ulong hi = n;
        if (!cand.empty()) {
            ulong top_j = cand.back();
            while (lo < hi) {
                ulong mid = lo + (hi - lo) / 2;
                if (value_at(new_j, mid) <= value_at(top_j, mid)) hi = mid;
                else lo = mid + 1;
            }
        }
        if (lo < n) {
            cand.push_back(new_j);
            start.push_back(lo);
        }
        while (front + 1 < cand.size() && start[front + 1] <= i) front += 1;

        bool have_real = !cand.empty() && start[front] <= i;
        double real_cost = have_real ? value_at(cand[front], i) : 0.0;
        double shortcut_cost = D_prev[i];
        if (have_real && real_cost <= shortcut_cost) {
            (*D_out)[i] = real_cost;
            (*T_out)[i] = cand[front];
        } else {
            (*D_out)[i] = shortcut_cost;
            (*T_out)[i] = i + 1;
        }
    }
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
    // * Set D and T using dynamic programming algorithm
    // ***************************************************

    // Algorithm as presented in section 2.2 of (Gronlund et al., 2017), with
    // SMAWK's matrix-search row-minimization replaced by Wilber's stack-based
    // technique (wilber_fill_row) for the same subproblem.

    CostCalculatorType cost_calculator(sorted_array, n, sort_idxs, args...);
    Matrix<double> D(k, n);
    Matrix<ulong> T(k, n);

    for (ulong i = 0; i < n; ++i) {
        D.set(0, i, cost_calculator.calc(0, i));
        T.set(0, i, 0);
    }

    for (ulong k_ = 1; k_ < k; ++k_) {
        vector<double> D_prev(n);
        for (ulong i = 0; i < n; ++i) D_prev[i] = D.get(k_ - 1, i);
        vector<double> D_row(n);
        vector<ulong> T_row(n);
        wilber_fill_row(n, D_prev, cost_calculator, &D_row, &T_row);
        for (ulong i = 0; i < n; ++i) {
            D.set(k_, i, D_row[i]);
            T.set(k_, i, T_row[i]);
        }
    }

    // ***************************************************
    // * Extract cluster assignments by backtracking
    // ***************************************************

    // TODO: This step requires O(kn) memory usage due to saving the entire
    //       T matrix. However, it can be modified so that the memory usage is O(n).
    //       D and T would not need to be retained in full (D already doesn't need
    //       to be fully retained, although it currently is).
    //       Details are in section 3 of (Grønlund et al., 2017).

    vector<double> sorted_clusters(n);

    ulong t = n;
    ulong k_ = k - 1;
    ulong n_ = n - 1;
    // The do/while loop was used in place of:
    //   for (k_ = k - 1; k_ >= 0; --k_)
    // to avoid wraparound of an unsigned type.
    do {
        ulong t_ = t;
        t = T.get(k_, n_);
        double centroid = 0.0;
        for (ulong i = t; i < t_; ++i) {
            sorted_clusters[i] = k_;
            // Mean computation: this is only for squared L2 cost calculators
            centroid += (
                (sorted_array[i] - centroid)
                * cost_calculator.weight(i, i)
                / cost_calculator.weight(t, i)
            );
        }
        centroids[k_] = centroid;
        k_ -= 1;
        n_ = t - 1;
    } while (t > 0);

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
