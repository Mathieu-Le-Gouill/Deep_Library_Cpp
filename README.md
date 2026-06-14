# Deep_Library_Cpp

A header-only C++23 deep learning library focused on performance, built from scratch without external dependencies. Demonstrates neural network fundamentals with hand-written SIMD acceleration and compile-time tensor shapes.

---

## Contents

### Tensor (`include/tensor/`)

The core data structure. `Tensor<Dims...>` is a variadic template class with statically-known dimensions checked at compile time.

**Arithmetic** — element-wise `+`, `-`, `*`, `/` with scalars and same-shape tensors, including rvalue overloads to avoid unnecessary copies.

**Matrix operations** — `mul`, `mul_transposed`, `transpose`, plus FMA-accelerated `multiply_and_add` / `multiply_and_sub` when the target CPU supports it.

**Initializers** — `zeros`, `ones`, `normal(mean, std)`, and weight initialization schemes: `glorot_normal`, `glorot_uniform`, `he_normal`, `he_uniform`, `lecun_normal`, `lecun_uniform`.

**Utilities** — `reshape`, `flatten`, `abs`, `sum`, `mean`, `variance`, `argmax`, `max`, `print`, `show_shape`.

**Activation helpers** — `apply_sigmoid`, `apply_ReLU` / `apply_ReLU_derivative`, `apply_Softmax` / `apply_Softmax_derivative`, `apply_normalization`, `norm_shift_and_scale`, `apply_dropout`.

### Layers (`include/layers/`)

All layers follow the CRTP base `Layer<Child, Input, Output>` with `Forward`, `Backward`, and `Update` methods.

| Layer | Description |
|---|---|
| `Dense<in, out, init>` | Fully-connected layer with SGD + momentum. |
| `Sigmoid<size>` | Element-wise sigmoid activation. |
| `ReLU<size>` | Element-wise ReLU activation. |
| `Softmax<size>` | Softmax + cross-entropy output layer. |
| `Flatten<W, H>` | Reshapes a 2-D input tensor into a 1-D vector. |
| `Norm<size>` | Batch normalisation layer. |
| `Dropout<size>` | Dropout regularisation layer. |
| `Conv<kernel, stride, ...>` | Convolutional layer (skeleton, not yet functional). |

### Pipeline (`include/pipeline/pipeline.h`)

`Pipeline<Layer1, Layer2, ...>` chains layers using a variadic tuple. `forward()` iterates in order; `backward()` iterates in reverse. `update()` calls `Update()` on every layer.

```cpp
Pipeline<
    Flatten<28, 28>,
    Dense<784, 32, LeCun_Normal>,
    Sigmoid<32>,
    Dense<32, 10, LeCun_Normal>,
    Sigmoid<10>
> network;

auto prediction = network.forward(input);
network.backward(error);
network.update();
```

### SIMD (`include/simd/`)

Single abstraction layer over SSE2 / SSE4 / AVX / AVX2 / FMA / AVX-512 using `#define` macros (e.g. `_ADD_PS`, `_FMADD`, `_LOAD`). At compile time the widest available instruction set is selected automatically. Includes portable `exp_ps` and `log_ps` approximations for activation functions.

### Compiler optimisations (`include/compiler/`)

`UNROLL_LOOP(n)` macro that maps to the correct `#pragma unroll` syntax for GCC, Clang, and ICC.

### Data loader (`include/loader/`)

`Load_MNIST_File` and `GetTargetValues` — read the MNIST binary image and label files into `Tensor` arrays ready for training.

### Network parameters (`include/network/`)

Centralised constants (`learningRate`, `momentum`, `epochs`, `minibatchSize`, `inputWidth/Height`, `outputSize`, …) with `static_assert` guards.

---

## Demo — MNIST digit classification

`src/main.cpp` trains a small MLP on the MNIST dataset and reports accuracy and loss on the test set.

Three compile-time flags at the top of the file control behaviour:

| Flag | Effect |
|---|---|
| `PERF_NET 1` | Print loss and accuracy after each epoch. |
| `TEST_NET 1` | Evaluate the model on the 10 000 test images after training. |
| `TIME_NET 1` | Print total training time in milliseconds. |

---

## Building and running

**Requirements**

- CMake ≥ 3.5
- A C++23-capable compiler (GCC ≥ 13 or Clang ≥ 17 recommended)
- A CPU with AVX2 + FMA support (Haswell / Zen 2 or newer)

**Steps**

```bash
# 1. Clone the repository
git clone <repo-url>
cd Deep_Libray_Cpp

# 2. Configure with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 3. Build
cmake --build build

# 4. Run (must be launched from the build/ directory so relative data paths resolve)
cd build
./Deep_Library_CPP
```

The binary looks for the MNIST files at `../data/` relative to its working directory, which is where the `data/` folder lives in this repository.

---

## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html)
- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010
- He et al., *Delving Deep into Rectifiers*, ICCV 2015
- LeCun et al., *Efficient BackProp*, Neural Networks: Tricks of the Trade, 1998
