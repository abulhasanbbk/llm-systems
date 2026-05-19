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