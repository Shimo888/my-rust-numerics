## Overview
This repository is a learning project for building a numerical computing library from scratch in Rust to understand its internal implementation.

## Contents
### 1: The Tensor Module
This is the core multi-dimensional array of the library.
- Zero-Copy Views:   
  It uses Arc and strides to perform shape manipulations like transposition (.transpose()) instantly, without copying memory. This is achieved by creating efficient "views" of the original data.
- Copy-on-Write (CoW):  
  When you try to modify an element of a view (a shared Tensor), the data is automatically copied. This prevents unintended side effects and ensures safe data manipulation.