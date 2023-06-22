#pragma once

#include <um2/config.hpp> // ENABLE_CUDA, HOSTDEV

#include <cstdio> // printf

// undef NDEBUG to enable asserts in release mode
#undef NDEBUG

// Overview:
// 1. Use TEST_CASE(name) to define a test case containing one or more 'assert'
// 2. Use MAKE_CUDA_KERNEL(name) to create a CUDA kernel from a test case, provided that
//      the test case was declared with HOSTDEV.
// 3. Use TEST_SUITE(name) to define a test suite containing one or more TEST(host_test)
//      or TEST_CUDA_KERNEL(host_test, blocks, threads).
//      - It is assumed MAKE_CUDA_KERNEL(host_test) was called before
//          TEST_CUDA_KERNEL(host_test, ...).
//      - TEST_CUDA_KERNEL(host_test) defaults to 1 block and 1 thread.
//      - TEST_CUDA_KERNEL(host_test, threads) defaults to 1 block.
// 4. Use RUN_TESTS(suite) to run a test suite in the main function.
//
// Additional notes:
// - TEST_HOSTDEV(name) is a shortcut for "TEST(name); TEST_CUDA_KERNEL(name)".

#define EXPECT_NEAR(a, b, eps) assert(!((a) < (b) - (eps)) && !((b) < (a) - (eps))) 

#define TEST_CASE(name) static void name()

#define TEST_SUITE(name) static void name()

#define TEST(name)                                                                       \
  printf("Running test case '%s'\n", #name);                                             \
  name();                                                                                \
  printf("Test case '%s' passed\n", #name);

#if UM2_ENABLE_CUDA

#  define MAKE_CUDA_KERNEL_1_ARGS(host_test)                                             \
    __global__ void host_test##_cuda_kernel() { host_test(); }

#  define MAKE_CUDA_KERNEL_2_ARGS(host_test, T)                                          \
    __global__ void host_test##_cuda_kernel() { host_test<T>(); }

#  define MAKE_CUDA_KERNEL_3_ARGS(host_test, T, U)                                       \
    __global__ void host_test##_cuda_kernel() { host_test<T, U>(); }

#  define MAKE_CUDA_KERNEL_4_ARGS(host_test, T, U, V)                                    \
    __global__ void host_test##_cuda_kernel() { host_test<T, U, V>(); }

#  define MAKE_CUDA_KERNEL_GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#  define MAKE_CUDA_KERNEL(...)                                                          \
    MAKE_CUDA_KERNEL_GET_MACRO(__VA_ARGS__, MAKE_CUDA_KERNEL_4_ARGS,                     \
                               MAKE_CUDA_KERNEL_3_ARGS, MAKE_CUDA_KERNEL_2_ARGS,         \
                               MAKE_CUDA_KERNEL_1_ARGS)                                  \
    (__VA_ARGS__)

#  define CUDA_KERNEL_POST_TEST                                                          \
    cudaDeviceSynchronize();                                                             \
    cudaError_t error = cudaGetLastError();                                              \
    if (error != cudaSuccess) {                                                          \
      printf("CUDA error: %s\n", cudaGetErrorString(error));                             \
      fflush(stdout);                                                                    \
      exit(1);                                                                           \
    }

#  define __TEST_CUDA_KERNEL(host_test, blocks, threads)                                 \
    {                                                                                    \
      printf("Running CUDA test case '%s' with %d blocks and %d threads\n", #host_test,  \
             blocks, threads);                                                           \
      host_test##_cuda_kernel<<<(blocks), (threads)>>>();                                \
      CUDA_KERNEL_POST_TEST                                                              \
      printf("CUDA test case '%s' finished\n", #host_test);                              \
    }

#  define __TEST_1TEMPLATE_CUDA_KERNEL(host_test, blocks, threads, T)                    \
    {                                                                                    \
      printf("Running CUDA test case '%s<%s>' with %d blocks and %d threads\n",          \
             #host_test, #T, blocks, threads);                                           \
      host_test##_cuda_kernel<T><<<(blocks), (threads)>>>();                             \
      CUDA_KERNEL_POST_TEST                                                              \
      printf("CUDA test case '%s<%s>' finished\n", #host_test, #T);                      \
    }

#  define __TEST_2TEMPLATE_CUDA_KERNEL(host_test, blocks, threads, T, U)                 \
    {                                                                                    \
      printf("Running CUDA test case '%s<%s, %s>' with %d blocks and %d threads\n",      \
             #host_test, #T, #U, blocks, threads);                                       \
      host_test##_cuda_kernel<T, U><<<(blocks), (threads)>>>();                          \
      CUDA_KERNEL_POST_TEST                                                              \
      printf("CUDA test case '%s<%s, %s>' finished\n", #host_test, #T, #U);              \
    }

#  define __TEST_3TEMPLATE_CUDA_KERNEL(host_test, blocks, threads, T, U, V)              \
    {                                                                                    \
      printf("Running CUDA test case '%s<%s, %s, %s>' with %d blocks and %d threads\n",  \
             #host_test, #T, #U, #V, blocks, threads);                                   \
      host_test##_cuda_kernel<T, U, V><<<(blocks), (threads)>>>();                       \
      CUDA_KERNEL_POST_TEST                                                              \
      printf("CUDA test case '%s<%s, %s, %s>' finished\n", #host_test, #T, #U, #V);      \
    }

#  define TEST_CUDA_KERNEL_1_ARGS(host_test) __TEST_CUDA_KERNEL(host_test, 1, 1)

#  define TEST_CUDA_KERNEL_2_ARGS(host_test, threads)                                    \
    __TEST_CUDA_KERNEL(host_test, 1, threads)

#  define TEST_CUDA_KERNEL_3_ARGS(host_test, blocks, threads)                            \
    __TEST_CUDA_KERNEL(host_test, blocks, threads)

#  define TEST_CUDA_KERNEL_4_ARGS(host_test, blocks, threads, T)                         \
    __TEST_1TEMPLATE_CUDA_KERNEL(host_test, blocks, threads, T)

#  define TEST_CUDA_KERNEL_5_ARGS(host_test, blocks, threads, T, U)                      \
    __TEST_2TEMPLATE_CUDA_KERNEL(host_test, blocks, threads, T, U)

#  define TEST_CUDA_KERNEL_6_ARGS(host_test, blocks, threads, T, U, V)                   \
    __TEST_3TEMPLATE_CUDA_KERNEL(host_test, blocks, threads, T, U, V)

#  define TEST_CUDA_KERNEL_GET_MACRO(_1, _2, _3, _4, _5, _6, NAME, ...) NAME
#  define TEST_CUDA_KERNEL(...)                                                          \
    TEST_CUDA_KERNEL_GET_MACRO(__VA_ARGS__, TEST_CUDA_KERNEL_6_ARGS,                     \
                               TEST_CUDA_KERNEL_5_ARGS, TEST_CUDA_KERNEL_4_ARGS,         \
                               TEST_CUDA_KERNEL_3_ARGS, TEST_CUDA_KERNEL_2_ARGS,         \
                               TEST_CUDA_KERNEL_1_ARGS)                                  \
    (__VA_ARGS__)

#else
#  define MAKE_CUDA_KERNEL(...)
#  define TEST_CUDA_KERNEL(...)
#endif

#define RUN_TESTS(suite)                                                                 \
  printf("Running test suite '%s'\n", #suite);                                           \
  suite();                                                                               \
  printf("Test suite '%s' passed\n", #suite);

#define TEST_HOSTDEV_1_ARGS(host_test)                                                   \
  TEST(host_test);                                                                       \
  TEST_CUDA_KERNEL(host_test);

#define TEST_HOSTDEV_2_ARGS(host_test, threads)                                          \
  TEST(host_test);                                                                       \
  TEST_CUDA_KERNEL(host_test, threads);

#define TEST_HOSTDEV_3_ARGS(host_test, blocks, threads)                                  \
  TEST(host_test);                                                                       \
  TEST_CUDA_KERNEL(host_test, blocks, threads);

// NOLINTBEGIN(bugprone-macro-parentheses)
#define TEST_HOSTDEV_4_ARGS(host_test, blocks, threads, T)                               \
  TEST(host_test<T>);                                                                    \
  TEST_CUDA_KERNEL(host_test, blocks, threads, T);

#define TEST_HOSTDEV_5_ARGS(host_test, blocks, threads, T, U)                            \
  TEST((host_test<T, U>));                                                               \
  TEST_CUDA_KERNEL(host_test, blocks, threads, T, U);

#define TEST_HOSTDEV_6_ARGS(host_test, blocks, threads, T, U, V)                         \
  TEST((host_test<T, U, V>));                                                            \
  TEST_CUDA_KERNEL(host_test, blocks, threads, T, U, V);
// NOLINTEND(bugprone-macro-parentheses)

#define TEST_HOSTDEV_GET_MACRO(_1, _2, _3, _4, _5, _6, NAME, ...) NAME
#define TEST_HOSTDEV(...)                                                                \
  TEST_HOSTDEV_GET_MACRO(__VA_ARGS__, TEST_HOSTDEV_6_ARGS, TEST_HOSTDEV_5_ARGS,          \
                         TEST_HOSTDEV_4_ARGS, TEST_HOSTDEV_3_ARGS, TEST_HOSTDEV_2_ARGS,  \
                         TEST_HOSTDEV_1_ARGS)                                            \
  (__VA_ARGS__)
