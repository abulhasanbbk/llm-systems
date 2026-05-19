# Chapter 1 - Matrix Multiplication and the First Matrix Class

## 1. Why Matrix Multiplication Matters in Deep Learning

Modern deep learning is built almost entirely on matrix operations.

A neural network layer is mainly a matrix multiplication:

Y = XW + b

Transformers, attention, and large language models all repeatedly perform matrix multiplication on very large tensors.

For example:

- embeddings are matrix lookups
- attention uses matrix multiplication between Q and K
- feed-forward layers are matrix multiplications
- MoE experts are matrix multiplications
- quantized inference still performs matrix multiplication

Because of this, understanding matrix multiplication at the systems level is essential for:
- transformers
- quantization
- FlashAttention
- GPU kernels
- inference optimisation

---

# 2. Representing a Matrix in Memory

In our implementation, a matrix is stored as a one-dimensional array:

```cpp
std::vector<float> data;
```
Even though we think of a matrix as 2D:
```text
1 2
3 4
```
Even though we think of a matrix as 2D:
```text
1 2
3 4
```
the computer stores it linearly in memory:
```text
[1, 2, 3, 4]
```
This is called row-major storage.

The element at position:
```text
(row, col)
```
is accessed using:
```cpp
data[row * cols + col]
```

This formula converts 2D coordinates into a 1D memory location.

This is one of the most important ideas in deep learning systems because tensors inside frameworks like PyTorch and TensorFlow are ultimately stored as contiguous memory buffers.

# 3. The Matrix Class

Our first abstraction is the `Matrix` class.

The class contains:

```cpp
int rows;
int cols;
std::vector<float> data;
```

The responsibilities of the class are:

- storing matrix dimensions
- storing matrix values
- accessing elements
- modifying elements
- printing matrices
- multiplying matrices

This transforms our code from a simple procedural program into a reusable numerical computing component.

# 4. Constructor

The constructor initializes the matrix dimensions and allocates memory.

```cpp
Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0f) {}
```

Explanation:

- `rows * cols` allocates enough memory for all elements
- each element is initialized to `0.0f`

For a 2×2 matrix:

```text
rows = 2
cols = 2
```

memory size becomes:

```text
2 × 2 = 4 float values
```

# 5. Setting Values

The `set()` function modifies a matrix element.

```cpp
void Matrix::set(int row, int col, float value) {
    data[row * cols + col] = value;
}
```

Example:

```cpp
A.set(0, 1, 2);
```

This stores the value `2` at:

- row 0
- column 1

Internally:

```text
index = row * cols + col
index = 0 * 2 + 1 = 1
```

So the value is stored at:

```text
data[1]
```

# 6. Getting Values

The `get()` function retrieves a matrix element.

```cpp
float Matrix::get(int row, int col) const {
    return data[row * cols + col];
}
```

The keyword `const` means the function does not modify the object.

This is important in C++ because it helps create safer and more predictable code.

# 7. Printing the Matrix

The `print()` function displays matrix values.

```cpp
void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << get(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
```

This introduces nested loops:

- outer loop iterates through rows
- inner loop iterates through columns

# 8. Matrix Multiplication

Matrix multiplication is the core operation of deep learning.

For matrices:

```text
A = (m × n)
B = (n × p)
```

the result becomes:

```text
C = (m × p)
```

The multiplication formula is:

```text
Cij = Σk Aik * Bkj
```

Our implementation:

```cpp
for (int i = 0; i < rows; i++) {

    for (int j = 0; j < other.cols; j++) {

        float sum = 0.0f;

        for (int k = 0; k < cols; k++) {
            sum += get(i, k) * other.get(k, j);
        }

        result.set(i, j, sum);
    }
}
```

Explanation:

- `i` iterates over rows of A
- `j` iterates over columns of B
- `k` performs the dot product

This is the classical triple-loop matrix multiplication algorithm.

# 9. Computational Complexity

Naive matrix multiplication has complexity:

```text
O(n^3)
```

This becomes extremely expensive for large matrices used in transformers.

For this reason, modern AI systems optimize matrix multiplication using:

- SIMD instructions
- GPU kernels
- tensor cores
- tiling
- quantization
- FlashAttention-style memory optimization

# 10. Benchmarking Matrix Multiplication

After implementing matrix multiplication, we need to measure how fast it runs.

This is called benchmarking.

Benchmarking is important because deep learning systems are not only about correctness. They are also about performance.

A matrix multiplication can give the correct result but still be too slow for real transformer inference.

In our code, we use the C++ `<chrono>` library to measure execution time.

```cpp
#include <chrono>
```

We create two random matrices:

```cpp
Matrix X(300, 300);
Matrix Y(300, 300);

X.randomize(-1.0f, 1.0f);
Y.randomize(-1.0f, 1.0f);
```

Then we record the time before multiplication:

```cpp
auto start = std::chrono::high_resolution_clock::now();
```

Next, we run matrix multiplication:

```cpp
Matrix Z = X.multiply(Y);
```

Then we record the time after multiplication:

```cpp
auto end = std::chrono::high_resolution_clock::now();
```

The elapsed time is computed using:

```cpp
std::chrono::duration<double> elapsed = end - start;
```

Finally, we print the result:

```cpp
std::cout << "Time for 300x300 matrix multiplication: "
          << elapsed.count()
          << " seconds"
          << std::endl;
```

This gives us the time taken by our matrix multiplication function.

For example, the output may look like:

```text
Time for 300x300 matrix multiplication: 0.035 seconds
```

The exact value depends on the computer, compiler, and optimisation level.

When compiling benchmark code, we should use compiler optimisation:

```bat
g++ -O2 cpp/chap01/main.cpp cpp/chap01/matrix.cpp -o cpp/chap01/main.exe
```

The `-O2` flag tells the compiler to optimise the code.

This is important because unoptimised C++ code can be much slower.

Common optimisation flags:
```bat
-O0   no optimisation, best for debugging
-O1   basic optimisation
-O2   strong optimisation, commonly used
-O3   more aggressive optimisation
-Ofast very aggressive, may change floating-point behaviour
```
We can benchmark using different flags by producing flag specific `.exe` files :
```bat
g++ -O0 ...
g++ -O2 ...
g++ -O3 ...
```
The following is the benchmarking result for a 300X300 matrix multiplication using different optimisaiton flags:
 
```text
| Compiler flag | Time for 300x300 multiplication |
|---|---|
| -O0 | 0.228882 seconds |
| -O2 | 0.016931  seconds |
| -O3 | 0.0167616 seconds |
```

For more on optimisation flags see https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html


Benchmarking also helps us compare different implementations, such as:

- naive matrix multiplication
- optimised matrix multiplication
- blocked matrix multiplication
- quantized matrix multiplication
- GPU matrix multiplication

This is the beginning of performance engineering.

In deep learning systems research, benchmarking is essential because modern models are limited not only by mathematics, but also by:

- memory access
- cache usage
- CPU speed
- GPU speed
- numerical precision
- data movement

This is why matrix multiplication is the right starting point for studying transformers, quantization, and efficient inference.

# 11. Connection to Transformers

Transformers repeatedly perform operations like:

```text
QK^T
```

and:

```text
Attention(Q,K,V) = softmax((QK^T) / sqrt(d))V
```

These are fundamentally matrix multiplications.

This means:

- understanding matrix multiplication
- understanding memory layout
- understanding numerical computation

are the true foundations of transformer systems research.

# 12. What We Have Built

At this stage, we now have:

- a reusable matrix abstraction
- manual memory layout understanding
- matrix indexing logic
- nested-loop numerical computation
- a basic matrix multiplication kernel

This is the first step toward:

- tensor libraries
- neural networks
- autograd systems
- transformers
- quantization engines
- inference runtimes like `llama.cpp`