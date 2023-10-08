To simplify the creation of CUDA kernels, we want certain data structures and
functions to be available in both HOST and DEVICE contexts. This includes many
of the structures and functions in the C++ standard library.

However, most of the standard library is HOST only, so we must write our own.
Most of structures and functions here are similar to [LLVM](https://github.com/llvm/llvm-project/tree/main),
but with a 32-bit signed integer index.
