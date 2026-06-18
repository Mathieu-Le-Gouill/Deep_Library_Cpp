// benchmarks/benchmark.cpp
//
// Compares this library's AVX2-vectorized kernels against scalar baselines
// on sizes matching the actual MNIST network:
//   Flatten<28,28> → Dense<784,32> → Sigmoid → Dense<32,10> → Sigmoid
//
// Build:  cd build && cmake .. && make benchmark
// Run:    ./benchmark

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <immintrin.h>
#include <cmath>

#include "tensor/tensor.h"
#include "network/network_parameters.h"

// ============================================================
// Scalar baselines — auto-vectorisation explicitly disabled
// ============================================================

#pragma GCC push_options
#pragma GCC optimize("O3,no-tree-vectorize")

// In-place element-wise add: a[i] += b[i]
__attribute__((noinline))
static void naive_add_ip(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] += b[i];
}

// In-place element-wise mul: a[i] *= b[i]
__attribute__((noinline))
static void naive_mul_ip(float* __restrict__ a, const float* __restrict__ b, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] *= b[i];
}

// Dense forward: W(rows×cols) · v(cols) → out(rows)
__attribute__((noinline))
static void naive_matvec(const float* __restrict__ W, const float* __restrict__ v,
                          float* __restrict__ out, size_t cols, size_t rows) {
    for (size_t r = 0; r < rows; ++r) {
        float s = 0.f;
        const float* row = W + r * cols;
        for (size_t c = 0; c < cols; ++c) s += row[c] * v[c];
        out[r] = s;
    }
}

// Outer product: a(m) ⊗ b(n) → out(m×n)
__attribute__((noinline))
static void naive_outer(const float* __restrict__ a, size_t m,
                         const float* __restrict__ b, size_t n,
                         float* __restrict__ out) {
    for (size_t i = 0; i < m; ++i) {
        const float ai = a[i];
        for (size_t j = 0; j < n; ++j) out[i * n + j] = ai * b[j];
    }
}

// ReLU: out[i] = max(0, a[i])
__attribute__((noinline))
static void naive_relu(float* __restrict__ a, size_t n) {
    for (size_t i = 0; i < n; ++i) a[i] = a[i] > 0.f ? a[i] : 0.f;
}

// Sigmoid: a[i] = 1 / (1 + exp(-a[i]))
__attribute__((noinline))
static void naive_sigmoid(float* __restrict__ a, size_t n) {
    for (size_t i = 0; i < n; ++i)
        a[i] = 1.f / (1.f + expf(-a[i]));
}

#pragma GCC pop_options

// ============================================================
// Timing: returns µs / call, averaged over `iters` iterations
// ============================================================

template<typename F>
double bench_us(F&& fn, int warmup = 200, int iters = 10000) {
    for (int i = 0; i < warmup; ++i) fn();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
}

// ============================================================
// Output helpers
// ============================================================

static void print_header() {
    std::cout << "\n"
              << std::left  << std::setw(44) << "Operation"
              << std::setw(26) << "Dimensions"
              << std::right << std::setw(11) << "Naive"
              << std::setw(13) << "SIMD (AVX2)"
              << std::setw(9)  << "Speedup"
              << "\n"
              << std::string(103, '-') << "\n";
}

static void print_row(const char* op, const char* dims,
                      double naive_us, double simd_us) {
    std::cout << std::left  << std::setw(44) << op
              << std::setw(26) << dims
              << std::right
              << std::setw(9)  << std::fixed << std::setprecision(3) << naive_us  << " µs"
              << std::setw(11) << std::fixed << std::setprecision(3) << simd_us   << " µs"
              << std::setw(8)  << std::fixed << std::setprecision(2) << (naive_us / simd_us) << "x"
              << "\n";
}

static void print_note(const char* msg) {
    std::cout << "  " << msg << "\n";
}

// ============================================================
// Main
// ============================================================

int main()
{
    std::cout
        << "======================================================================\n"
        << "  Deep Library C++ — SIMD Kernel Benchmark\n"
        << "  CPU   : Intel Core i7-14700KF  (measured @ 5.5 GHz)\n"
        << "  SIMD  : AVX2 + FMA  (8 × float32 per register, 2 FMAs/cycle)\n"
        << "  Build : -O3 -mavx2 -mfma -msse4.1 -msse4.2  (C++23)\n"
        << "  Naive : scalar loops, same -O3, auto-vectorisation disabled\n"
        << "  Times : mean over 10 000 iterations (µs per call)\n"
        << "======================================================================\n";

    // Helper: allocate 32-byte-aligned buffer and fill with constant
    auto alloc_fill = [](size_t n, float val) -> float* {
        float* p = (float*)_mm_malloc(n * sizeof(float), 32);
        for (size_t i = 0; i < n; ++i) p[i] = val;
        return p;
    };

    // ============================================================
    // Section 1 – Element-wise operations  (in-place, no alloc)
    // ============================================================
    print_header();
    std::cout << "  [Element-wise — in-place (operator +=, *=), no allocation overhead]\n\n";

    // ---- 1a. elem add — 784 floats (one input vector) ----
    {
        constexpr size_t N = 784;
        float* a = alloc_fill(N, 0.5f);
        float* b = alloc_fill(N, 0.01f);
        Tensor<N> ta(a), tb(b);

        double naive_t = bench_us([&]{ naive_add_ip(a, b, N); });
        double simd_t  = bench_us([&]{ ta += tb; });
        print_row("Element-wise add  (+=)", "784 floats", naive_t, simd_t);
        _mm_free(a); _mm_free(b);
    }

    // ---- 1b. elem add — 784×32 floats (weight matrix) ----
    {
        constexpr size_t N = 784 * 32; // 25 088 floats = 98 KB
        float* a = alloc_fill(N, 0.5f);
        float* b = alloc_fill(N, 0.01f);
        Tensor<N> ta(a), tb(b);

        double naive_t = bench_us([&]{ naive_add_ip(a, b, N); });
        double simd_t  = bench_us([&]{ ta += tb; });
        print_row("Element-wise add  (+=)", "784×32 floats (98 KB)", naive_t, simd_t);
        _mm_free(a); _mm_free(b);
    }

    // ---- 1c. elem mul — 784 floats ----
    {
        constexpr size_t N = 784;
        float* a = alloc_fill(N, 0.999f);
        float* b = alloc_fill(N, 0.999f);
        Tensor<N> ta(a), tb(b);

        double naive_t = bench_us([&]{ naive_mul_ip(a, b, N); });
        double simd_t  = bench_us([&]{ ta *= tb; });
        print_row("Element-wise mul  (*=)", "784 floats", naive_t, simd_t);
        _mm_free(a); _mm_free(b);
    }

    // ---- 1d. elem mul — 784×32 floats ----
    {
        constexpr size_t N = 784 * 32;
        float* a = alloc_fill(N, 0.999f);
        float* b = alloc_fill(N, 0.999f);
        Tensor<N> ta(a), tb(b);

        double naive_t = bench_us([&]{ naive_mul_ip(a, b, N); });
        double simd_t  = bench_us([&]{ ta *= tb; });
        print_row("Element-wise mul  (*=)", "784×32 floats (98 KB)", naive_t, simd_t);
        _mm_free(a); _mm_free(b);
    }

    // ============================================================
    // Section 2 – Dense Forward  (mul_b_transposed_scalar)
    // Each call allocates the small output tensor internally.
    // ============================================================
    std::cout << "\n  [Dense Forward — W·x → out; includes small output alloc]\n\n";

    // ---- 2a. 784 → 32  (first layer) ----
    {
        constexpr size_t COLS = 784, ROWS = 32;
        constexpr size_t FLOPS = COLS * ROWS * 2; // 50 176

        float* W     = alloc_fill(ROWS * COLS, 0.1f);
        float* v     = alloc_fill(COLS, 0.5f);
        float* out_n = alloc_fill(ROWS, 0.f);

        Tensor<COLS, ROWS> tW(W);
        Tensor<COLS>       tv(v);

        double naive_t = bench_us([&]{
            naive_matvec(W, v, out_n, COLS, ROWS);
            volatile float s = out_n[0]; (void)s;
        });
        double simd_t = bench_us([&]{
            auto r = mul_b_transposed_scalar(tW, tv);
            volatile float s = r(0); (void)s;
        });

        char dims[64]; snprintf(dims, sizeof(dims), "784×32 · v(784)  [%zu FLOPs]", FLOPS);
        print_row("Dense Forward  (mat-vec)", dims, naive_t, simd_t);
        _mm_free(W); _mm_free(v); _mm_free(out_n);
    }

    // ---- 2b. 32 → 10  (second layer) ----
    {
        constexpr size_t COLS = 32, ROWS = 10;
        constexpr size_t FLOPS = COLS * ROWS * 2; // 640

        float* W     = alloc_fill(ROWS * COLS, 0.1f);
        float* v     = alloc_fill(COLS, 0.5f);
        float* out_n = alloc_fill(ROWS, 0.f);

        Tensor<COLS, ROWS> tW(W);
        Tensor<COLS>       tv(v);

        double naive_t = bench_us([&]{
            naive_matvec(W, v, out_n, COLS, ROWS);
            volatile float s = out_n[0]; (void)s;
        });
        double simd_t = bench_us([&]{
            auto r = mul_b_transposed_scalar(tW, tv);
            volatile float s = r(0); (void)s;
        });

        char dims[64]; snprintf(dims, sizeof(dims), "32×10 · v(32)    [%zu FLOPs]", FLOPS);
        print_row("Dense Forward  (mat-vec)", dims, naive_t, simd_t);
        _mm_free(W); _mm_free(v); _mm_free(out_n);
    }

    // ============================================================
    // Section 3 – Dense Backward  (mul_transposed_scalar = outer product)
    // Weight gradient accumulation.
    // ============================================================
    std::cout << "\n  [Dense Backward — grad ⊗ input → ΔW; includes output alloc]\n\n";

    // ---- 3a. grad(32) ⊗ input(784) → ΔW(32×784) ----
    {
        constexpr size_t M = 32, N = 784;
        constexpr size_t FLOPS = M * N * 2; // 50 176

        float* a     = alloc_fill(M, 0.1f);
        float* b     = alloc_fill(N, 0.5f);
        float* out_n = alloc_fill(M * N, 0.f);

        Tensor<M> ta(a);
        Tensor<N> tb(b);

        double naive_t = bench_us([&]{
            naive_outer(a, M, b, N, out_n);
            volatile float s = out_n[0]; (void)s;
        });
        double simd_t = bench_us([&]{
            auto r = mul_transposed_scalar(ta, tb);
            volatile float s = r(0); (void)s;
        });

        char dims[64]; snprintf(dims, sizeof(dims), "vec(32) ⊗ vec(784) [%zu FLOPs]", FLOPS);
        print_row("Dense Backward (outer product)", dims, naive_t, simd_t);
        _mm_free(a); _mm_free(b); _mm_free(out_n);
    }

    // ---- 3b. grad(10) ⊗ input(32) → ΔW(10×32) ----
    {
        constexpr size_t M = 10, N = 32;
        constexpr size_t FLOPS = M * N * 2; // 640

        float* a     = alloc_fill(M, 0.1f);
        float* b     = alloc_fill(N, 0.5f);
        float* out_n = alloc_fill(M * N, 0.f);

        Tensor<M> ta(a);
        Tensor<N> tb(b);

        double naive_t = bench_us([&]{
            naive_outer(a, M, b, N, out_n);
            volatile float s = out_n[0]; (void)s;
        });
        double simd_t = bench_us([&]{
            auto r = mul_transposed_scalar(ta, tb);
            volatile float s = r(0); (void)s;
        });

        char dims[64]; snprintf(dims, sizeof(dims), "vec(10) ⊗ vec(32)  [%zu FLOPs]", FLOPS);
        print_row("Dense Backward (outer product)", dims, naive_t, simd_t);
        _mm_free(a); _mm_free(b); _mm_free(out_n);
    }

    // ============================================================
    // Section 4 – Activations
    // ============================================================
    std::cout << "\n  [Activations — in-place, no allocation]\n\n";

    // ---- 4a. ReLU — 784 values ----
    {
        constexpr size_t N = 784;
        float* a = alloc_fill(N, 0.5f);
        Tensor<N> ta(a);

        double naive_t = bench_us([&]{ naive_relu(a, N); });
        double simd_t  = bench_us([&]{ ta.apply_ReLU(); });
        print_row("ReLU  (SIMD: _MAX_PS)", "784 values", naive_t, simd_t);
        _mm_free(a);
    }

    // ---- 4b. ReLU — 32 values ----
    {
        constexpr size_t N = 32;
        float* a = alloc_fill(N, 0.5f);
        Tensor<N> ta(a);

        double naive_t = bench_us([&]{ naive_relu(a, N); });
        double simd_t  = bench_us([&]{ ta.apply_ReLU(); });
        print_row("ReLU  (SIMD: _MAX_PS)", "32 values", naive_t, simd_t);
        _mm_free(a);
    }

    // ---- 4c. Sigmoid — currently scalar in the library ----
    {
        constexpr size_t N = 784;
        float* a = alloc_fill(N, 0.5f);
        Tensor<N> ta(a);

        double naive_t = bench_us([&]{ naive_sigmoid(a, N); });
        double simd_t  = bench_us([&]{ ta.apply_sigmoid(); });
        print_row("Sigmoid  (SIMD: exp_ps + DIV)", "784 values", naive_t, simd_t);
        _mm_free(a);
    }

    std::cout << "\n";
    return 0;
}
