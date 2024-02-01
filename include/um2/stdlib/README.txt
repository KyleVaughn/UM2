===============================================================================
OVERVIEW
===============================================================================
To facilitate the development of high performance code on both CPUs (host) and
GPUs (device), we provide an implementation of a subset of the C++ standard
library which is compatible with both CUDA and standard C++ compilers. This is
necessary because the majority of the standard library is not available for use
in device code, which severely limits our ability to write portable code that
can be executed on both CPUs and GPUs.

Drop-in Replacement
-------------------
The functions and classes defined here are intended to be used as drop-in
replacements for the standard library equivalents. As such, we choose not to
provide functions like "std::sort", which would likely be inappropriate for use
on a single thread of a GPU due to issues of thread divergence and memory
access patterns. However, we do provide functions like "um2::max", which is a
convenient replacement for "std::max" that works in kernel code.

Naming
------
UM2 uses camelCase for function names and PascalCase for class names, whereas
the standard library uses snake_case for both. To maintain some consistency,
we retain snake_case names for the standard library functions due to their
ubiquity, but use PascalCase for class names. 

It is important that we differentiate between the std:: and um2:: versions 
of the classes, since UM2 uses a 32-bit index type by default, whereas the 
standard library uses a 64-bit index type. This is because 32-bit arithmetic 
is substantially faster than 64-bit arithmetic on GPUs, and index calculations 
are common in many of the data structures used in UM2.
