#pragma once

#include <um2/common/config.hpp> // UM2_ENABLE_CUDA, UM2_HOSTDEV

#include <cmath>   // std::abs
#include <cstdio>  // printf
#include <cstdlib> // exit, EXIT_FAILURE

// A very simple GPU-compatible test framework.
//
// Overview:
// 1. Use EXPECT_XXX macros to check conditions for non-fatal errors.
//  - If a condition is not met, the test will fail, but execution will continue.
//  - EXPECT_TRUE(a)          to check that a is true
//  - EXPECT_FALSE(a)         to check that a is false
//  - EXPECT_EQ(a, b)         to check that a == b
//  - EXPECT_NE(a, b)         to check that a != b
//  - EXPECT_LT(a, b)         to check that a < b
//  - EXPECT_LE(a, b)         to check that a <= b
//  - EXPECT_GT(a, b)         to check that a > b
//  - EXPECT_GE(a, b)         to check that a >= b
//  - EXPECT_NEAR(a, b, eps)  to check that abs(a - b) < eps
//  ... more to come
// 2. Use ASSERT_XXX macros to check conditions for fatal errors.
//  - If a condition is not met, the test will fail and execution will stop.
//  ... more to come
// 3. Use TEST_CASE(name) to define a test case containing one or more EXPECT_XXX or
// ASSERT_XXX.
// 4. Use MAKE_CUDA_KERNEL(name) to create a CUDA kernel from a test case, provided that
// the test
//   case was declared with UM2_HOSTDEV.
// 5. Use TEST_SUITE(name) to define a test suite containing one or more TEST(host_test)
// or
//   TEST_CUDA_KERNEL(host_test, blocks, threads).
//   - It is assumed MAKE_CUDA_KERNEL(host_test) was called before
//   TEST_CUDA_KERNEL(host_test, ...).
//   - TEST_CUDA_KERNEL(host_test) defaults to 1 block and 1 thread.
//   - TEST_CUDA_KERNEL(host_test, threads) defaults to 1 block.
// 6. Use RUN_TESTS(suite) to run a test suite in the main function.
//
// Additional notes:
// - TEST_HOSTDEV(name) is a shortcut for TEST(name) and TEST_CUDA_KERNEL(name).

struct TestResult {
  int num_failures = 0;

  UM2_HOSTDEV void failure(char const * file, int line, char const * function,
                           char const * message)
  {
    num_failures++;
    printf("Test failed in %s:%d\n", file, line);
    printf("  Function: %s\n", function);
    printf("  Message: %s", message); // Purposefully no newline
  }
};

#define EXPECT_TRUE(a)                                                                   \
  if (!(a)) {                                                                            \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected " #a " to be true, but got false\n");                      \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define EXPECT_FALSE(a)                                                                  \
  if (a) {                                                                               \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected " #a " to be false, but got true\n");                      \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define EXPECT_EQ(a, b)                                                                  \
  if (!((a) == (b))) {                                                                   \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected " #a " == " #b ", but got " #a " != " #b "\n");            \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define EXPECT_NE(a, b)                                                                  \
  if (!((a) != (b))) {                                                                   \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected " #a " != " #b ", but got " #a " == " #b "\n");            \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define EXPECT_LT(a, b)                                                                  \
  if (!((a) < (b))) {                                                                    \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected " #a " < " #b ", but got " #a " >= " #b "\n");             \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define EXPECT_LE(a, b)                                                                  \
  if (!((a) <= (b))) {                                                                   \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected " #a " <= " #b ", but got " #a " > " #b "\n");             \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define EXPECT_GT(a, b)                                                                  \
  if (!((a) > (b))) {                                                                    \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected " #a " > " #b ", but got " #a " <= " #b "\n");             \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define EXPECT_GE(a, b)                                                                  \
  if (!((a) >= (b))) {                                                                   \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected " #a " >= " #b ", but got " #a " < " #b "\n");             \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define EXPECT_NEAR(a, b, eps)                                                           \
  if (std::abs((a) - (b)) > (eps)) {                                                     \
    result->failure(__FILE__, __LINE__, __FUNCTION__,                                    \
                    "Expected abs(" #a " - " #b ") < " #eps ", but got abs(" #a " - " #b \
                    ") = ");                                                             \
    printf("%f\n", static_cast<double>(std::abs((a) - (b))));                            \
    if (exit_on_failure) {                                                               \
      return;                                                                            \
    }                                                                                    \
  }

#define TEST_CASE(name) static void name(TestResult * const result, bool exit_on_failure)

#define TEST_SUITE(name) static void name(TestResult & result, bool exit_on_failure)

#define TEST(name)                                                                       \
  {                                                                                      \
    int const num_failures_before = result.num_failures;                                \
    printf("Running test case '%s'\n", #name);                                           \
    name(&result, exit_on_failure);                                                      \
    int const num_failures_after = result.num_failures - num_failures_before;           \
    printf("Test case '%s' finished with %d failures\n", #name, num_failures_after);     \
    if (result.num_failures > 0 && exit_on_failure) {                                    \
      return;                                                                            \
    }                                                                                    \
  }

#if UM2_ENABLE_CUDA

#  define MAKE_UNTEMPLATED_CUDA_KERNEL(host_test)                                        \
    __global__ void host_test##_cuda_kernel(TestResult * const result,                   \
                                            bool * const exit_on_failure)                \
    {                                                                                    \
      host_test(result, *exit_on_failure);                                               \
    }

#  define MAKE_1TEMPLATE_CUDA_KERNEL(host_test, T)                                       \
    __global__ void host_test##_cuda_kernel(TestResult * const result,                   \
                                            bool * const exit_on_failure)                \
    {                                                                                    \
      host_test<T>(result, *exit_on_failure);                                               \
    }

#  define MAKE_2TEMPLATE_CUDA_KERNEL(host_test, T, U)                                    \
    __global__ void host_test##_cuda_kernel(TestResult * const result,                   \
                                            bool * const exit_on_failure)                \
    {                                                                                    \
      host_test<T, U>(result, *exit_on_failure);                                         \
    }

#  define MAKE_CUDA_KERNEL_1_ARGS(host_test) MAKE_UNTEMPLATED_CUDA_KERNEL(host_test)

#  define MAKE_CUDA_KERNEL_2_ARGS(host_test, T)                                    \
    MAKE_1TEMPLATE_CUDA_KERNEL(host_test, T)

#  define MAKE_CUDA_KERNEL_3_ARGS(host_test, T, U)                                    \
    MAKE_2TEMPLATE_CUDA_KERNEL(host_test, T, U)

#  define MAKE_CUDA_KERNEL_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#  define MAKE_CUDA_KERNEL(...)                                                          \
    MAKE_CUDA_KERNEL_GET_MACRO(__VA_ARGS__, MAKE_CUDA_KERNEL_3_ARGS,                     \
                               MAKE_CUDA_KERNEL_2_ARGS, MAKE_CUDA_KERNEL_1_ARGS)         \
    (__VA_ARGS__)

#  define __TEST_CUDA_KERNEL_SETUP                                                       \
      int const num_failures_before = result.num_failures;                               \
      TestResult * device_result;                                                        \
      cudaMalloc(&device_result, sizeof(TestResult));                                    \
      cudaMemcpy(device_result, &result, sizeof(TestResult), cudaMemcpyHostToDevice);    \
      bool * device_exit_on_failure;                                                     \
      cudaMalloc(&device_exit_on_failure, sizeof(bool));                                 \
      cudaMemcpy(device_exit_on_failure, &exit_on_failure, sizeof(bool),                 \
                 cudaMemcpyHostToDevice);

# define __TEST_CUDA_KERNEL_POST                                                         \
      cudaDeviceSynchronize();                                                           \
      cudaError_t error = cudaGetLastError();                                            \
      if (error != cudaSuccess) {                                                        \
        printf("CUDA error: %s\n", cudaGetErrorString(error));                           \
        return;                                                                          \
      }                                                                                  \
      cudaMemcpy(&result, device_result, sizeof(TestResult), cudaMemcpyDeviceToHost);    \
    int const num_failures_after = result.num_failures - num_failures_before;            \

#  define __TEST_CUDA_KERNEL(host_test, blocks, threads)                                 \
    {                                                                                    \
      __TEST_CUDA_KERNEL_SETUP                                                           \
      printf("Running CUDA test case '%s' with %d blocks and %d threads\n", #host_test,  \
             blocks, threads);                                                           \
      host_test##_cuda_kernel<<<(blocks), (threads)>>>(device_result,                    \
                                                       device_exit_on_failure);          \
      __TEST_CUDA_KERNEL_POST                                                            \
      printf("CUDA test case '%s' finished with %d failures\n", #host_test,              \
             num_failures_after);                                                        \
      if (result.num_failures > 0 && exit_on_failure) {                                  \
        return;                                                                          \
      }                                                                                  \
    }

#  define __TEST_1TEMPLATE_CUDA_KERNEL(host_test, blocks, threads, T)                    \
    {                                                                                    \
      __TEST_CUDA_KERNEL_SETUP                                                           \
      printf("Running CUDA test case '%s'<%s> with %d blocks and %d threads\n",          \
             #host_test, #T, blocks, threads);                                           \
      host_test##_cuda_kernel<T><<<(blocks), (threads)>>>(device_result,                 \
                                                       device_exit_on_failure);          \
      __TEST_CUDA_KERNEL_POST                                                            \
      cudaMemcpy(&result, device_result, sizeof(TestResult), cudaMemcpyDeviceToHost);    \
      printf("CUDA test case '%s'<%s> finished with %d failures\n", #host_test,          \
            #T, num_failures_after);                                                     \
      if (result.num_failures > 0) {                                                     \
        return;                                                                          \
      }                                                                                  \
    }

#  define TEST_CUDA_KERNEL_1_ARGS(host_test) __TEST_CUDA_KERNEL(host_test, 1, 1)

#  define TEST_CUDA_KERNEL_2_ARGS(host_test, threads)                                    \
    __TEST_CUDA_KERNEL(host_test, 1, threads)

#  define TEST_CUDA_KERNEL_3_ARGS(host_test, blocks, threads)                            \
    __TEST_CUDA_KERNEL(host_test, blocks, threads)

#  define TEST_CUDA_KERNEL_4_ARGS(host_test, blocks, threads, T)                         \
    __TEST_1TEMPLATE_CUDA_KERNEL(host_test, blocks, threads, T)

#  define TEST_CUDA_KERNEL_GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#  define TEST_CUDA_KERNEL(...)                                                          \
    TEST_CUDA_KERNEL_GET_MACRO(__VA_ARGS__, TEST_CUDA_KERNEL_4_ARGS,                     \
                                            TEST_CUDA_KERNEL_3_ARGS,                     \
                                            TEST_CUDA_KERNEL_2_ARGS,                     \
                                            TEST_CUDA_KERNEL_1_ARGS)                     \
    (__VA_ARGS__)

#else
#  define MAKE_CUDA_KERNEL(...)
#  define TEST_CUDA_KERNEL(...)
#endif

#define RUN_THE_TESTS(suite, exit_on_failure)                                            \
  {                                                                                      \
    TestResult result;                                                                   \
    printf("Running test suite '%s'\n", #suite);                                         \
    suite(result, exit_on_failure);                                                      \
    printf("Test suite '%s' finished with %d failures\n", #suite, result.num_failures);  \
    if (result.num_failures > 0) {                                                       \
      return 1;                                                                          \
    }                                                                                    \
  }

#define RUN_TESTS_1_ARGS(suite) RUN_THE_TESTS(suite, true)

#define RUN_TESTS_2_ARGS(suite, exit_on_failure) RUN_THE_TESTS(suite, exit_on_failure)

#define RUN_TESTS_GET_MACRO(_1, _2, NAME, ...) NAME
#define RUN_TESTS(...)                                                                   \
  RUN_TESTS_GET_MACRO(__VA_ARGS__, RUN_TESTS_2_ARGS, RUN_TESTS_1_ARGS)(__VA_ARGS__)

#define TEST_HOSTDEV_1_ARGS(host_test)                                               \
  TEST(host_test);                                                                            \
  TEST_CUDA_KERNEL(host_test);

#define TEST_HOSTDEV_2_ARGS(host_test, threads)                                    \
  TEST(host_test);                                                                            \
  TEST_CUDA_KERNEL(host_test, threads);

#define TEST_HOSTDEV_3_ARGS(host_test, blocks, threads)                            \
  TEST(host_test);                                                                            \
  TEST_CUDA_KERNEL(host_test, blocks, threads);

#define TEST_HOSTDEV_4_ARGS(host_test, blocks, threads, T)                         \
  TEST(host_test<T>);                                                                      \
  TEST_CUDA_KERNEL(host_test, blocks, threads, T);

#define TEST_HOSTDEV_GET_MACRO(_1, _2, _3, _4, NAME, ...) NAME
#define TEST_HOSTDEV(...)                                                          \
  TEST_HOSTDEV_GET_MACRO(__VA_ARGS__, TEST_HOSTDEV_4_ARGS,                     \
                                      TEST_HOSTDEV_3_ARGS,                     \
                                      TEST_HOSTDEV_2_ARGS,                     \
                                      TEST_HOSTDEV_1_ARGS)                     \
  (__VA_ARGS__)
