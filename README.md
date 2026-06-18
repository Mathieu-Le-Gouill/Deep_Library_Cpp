# Deep_Library_Cpp

**A header-only C++23 deep-learning library built from scratch.**

![build](https://img.shields.io/badge/build-passing-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C)
![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%2B%20FMA-orange)

`Tensor<Dims...>` carries its shape in the type system, every loop is statically sized, and the best available instruction set (SSE2 → AVX-512) is selected at compile time. No external dependencies and no runtime dispatch.

---

## Benchmarks

> **CPU:** Intel Core i7-14700KF · AVX2 + FMA  
> **Build:** `-O3 -mavx2 -mfma`, C++23, GCC 13, single thread  
> **Method:** minimum over 25 batches × 20 000 calls, on byte-identical input

**Scalar** is the same code compiled without auto-vectorisation, it isolates what SIMD actually buys. 

**Eigen 3.4** (fixed-size, header-only) is the external reference. Ratios **> 1** mean this library is faster.

| Operation | Size | Scalar (µs) | This – AVX2 (µs) | Eigen (µs) | vs Scalar | vs Eigen |
|---|---|---:|---:|---:|---:|---:|
| Element-wise `+=` | 784 floats | 0.152 | 0.018 | 0.025 | **8.5×** | **1.4×** |
| Element-wise `+=` | 784 × 32 (98 KB) | 5.35 | 1.28 | 1.29 | **4.2×** | 1.0× |
| Element-wise `*=` | 784 floats | 0.149 | 0.020 | 0.027 | **7.7×** | **1.4×** |
| Element-wise `*=` | 784 × 32 (98 KB) | 5.41 | 1.30 | 1.36 | **4.2×** | 1.1× |
| Dense forward W·x | 784 × 32 → 32 | 8.68 | 0.519 | 0.519 | **16.7×** | 1.0× |
| Dense forward W·x | 32 × 10 → 10 | 0.071 | 0.013 | 0.016 | 5.2× | 1.2× |
| Outer product ⊗ | vec(32) ⊗ vec(784) | 4.79 | 1.15 | 1.16 | **4.2×** | 1.0× |
| ReLU | 784 values | 0.200 | 0.014 | 0.023 | **14.5×** | **1.7×** |
| Sigmoid † | 784 values | 1.20 | 0.237 | 0.324 | **5.1×** | **1.4×** |

† *Uses a fast polynomial `exp` (approximate); Eigen uses an accurate vectorised `exp`.*

The hand-written kernels deliver **4–16× over scalar**. 

Against Eigen, the library is now competitive across every operation: matching exactly on the compute-bound mat-vec pass, within noise on memory-bandwidth-bound ops, and modestly faster on small element-wise and activation ops.

**Reproduce:**
```bash
cmake -B build && cmake --build build --target benchmark
cd build && ./benchmark
```

---

## Quick Start

```cpp
#include "pipeline/pipeline.h"

// A 784 → 32 → 10 MLP for MNIST, fully specified at compile time.
Pipeline<
    Flatten<28, 28>,
    Dense<784, 32, LeCun_Normal>,
    Sigmoid<32>,
    Dense<32, 10, LeCun_Normal>,
    Sigmoid<10>
> network;

Tensor<10> prediction = network.forward(input);
Tensor<10> error      = prediction - target;
network.backward(error);
network.update();
```

Shape mismatches are **build errors**, not runtime crashes. A complete MNIST training loop lives in [`src/main.cpp`](src/main.cpp).

---

## Building

**Requirements:** CMake ≥ 3.5 · GCC ≥ 13 or Clang ≥ 17 · AVX2 + FMA (Haswell / Zen 2+)

```bash
git clone <repo-url> && cd Deep_Libray_Cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --build build --target benchmark

cd build && ./Deep_Library_CPP   # MNIST demo
./benchmark                      # kernel benchmarks
```

---

## Architecture

Everything is hand-written. Eigen is vendored under `third_party/` solely as a benchmark reference.

### Tensors - `include/tensor/`

`Tensor<Dims...>` is a variadic template whose dimensions live entirely in the type system. All sizes, strides, and loop bounds are `constexpr`, enabling full unrolling and vectorisation with zero runtime bookkeeping. Supports element-wise arithmetic, `mul` / `mul_transposed` / `transpose`, FMA-fused operations, reductions (`sum`, `mean`, `argmax`), and weight initialisers (`glorot_*`, `he_*`, `lecun_*`).

### SIMD - `include/simd/`

A macro layer over SSE2 / SSE4 / AVX / AVX2 / FMA / AVX-512. The widest ISA available on the target is selected at compile time - same source, best output. Includes portable polynomial `exp_ps` / `log_ps` approximations.

### Layers - `include/layers/`

CRTP base `Layer<Child, Input, Output>` with zero virtual dispatch.

| Layer | Description |
|---|---|
| `Dense<in, out, init>` | Fully-connected · SGD + momentum |
| `Sigmoid` / `ReLU` / `Softmax` | Activations (`Softmax` fused with cross-entropy) |
| `Flatten<W, H>` | 2-D → 1-D reshape |
| `Norm<size>` | Batch normalisation |
| `Dropout<size>` | Dropout regularisation |

### Pipeline - `include/pipeline/`

`Pipeline<Layers...>` chains layers through a variadic tuple. `forward()`, `backward()`, and `update()` are fully resolved at compile time.

---

## Roadmap

- **Convolutions**: `Conv` is declared but not yet functional; the next major piece.
- **Multi-threading**: kernels are tuned for a single core; batched/tiled GEMM would extend the story to larger layers.
- **More optimisers**: only SGD + momentum today; Adam / RMSProp planned.
- **Dynamic graphs**: compile-time shapes are great for safety and inlining, but fix topology at build time.

---

## References

- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010
- He et al., *Delving Deep into Rectifiers*, ICCV 2015
- LeCun et al., *Efficient BackProp*, Neural Networks: Tricks of the Trade, 1998
