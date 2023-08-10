In order to allow for maximium code reuse, we need certain data structures and
functions to be available in both HOST and DEVICE contexts.

Many of the structures and functions we need are available in the C++ standard library,
but not in CUDA.

The code in this directory is intended to be used as a HOST/DEVICE agnostic implementation
of the standard library, although it is not a complete implementation.

When CUDA is not enabled, many functions are simply aliases to the standard library.

Note that UM2 uses a 32-bit signed integer for indexing by default, so we do not 
alias std::vector or std::string, which use 64-bit unsigned integers for indexing.
