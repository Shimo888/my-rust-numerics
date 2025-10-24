## Overview
This repository is a learning project for building a numerical computing library from scratch in Rust to understand its internal implementation.

## Contents
### 1: The Tensor Module
This is the core multi-dimensional array of the library.
- Zero-Copy Views:   
  It uses Arc and strides to perform shape manipulations like transposition (.transpose()) instantly, without copying memory. This is achieved by creating efficient "views" of the original data.
- Copy-on-Write (CoW):  
  When you try to modify an element of a view (a shared Tensor), the data is automatically copied. This prevents unintended side effects and ensures safe data manipulation.
- Implementation via Composition of High-Level Operations:  
  Instead of using slow, nested loops—which cause inefficient, non-sequential memory access and frequent CPU cache misses—complex operations like tensor contraction (tensor_dot) are achieved by composing fundamental, high-performance building blocks.
  This strategy (maybe) transforms the problem into a cache-friendly form by using axis permutation (permute), reshaping into a matrix (reshape), and finally, optimized matrix multiplication (mat_mul). This allows the operation to maximize memory throughput for significant performance gains.