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
//  - EXPECT_NEAR(a, b, eps)  to check that abs(a - b) < eps
//  ... more to come
// 2. Use ASSERT_XXX macros to check conditions for fatal errors.
//  - If a condition is not met, the test will fail and execution will stop.
//  ... more to come
// 3. Use TEST_CASE(name) to define a test case containing one or more EXPECT_XXX or ASSERT_XXX.
// 4. Use MAKE_CUDA_KERNEL(name) to create a CUDA kernel from a test case, provided that the test
//   case was declared with UM2_HOSTDEV.
// 5. Use TEST_SUITE(name) to define a test suite containing one or more TEST(host_test) or
//   TEST_CUDA_KERNEL(host_test, blocks, threads).
//   - It is assumed MAKE_CUDA_KERNEL(host_test) was called before TEST_CUDA_KERNEL(host_test, ...).
//   - TEST_CUDA_KERNEL(host_test) defaults to 1 block and 1 thread.
//   - TEST_CUDA_KERNEL(host_test, threads) defaults to 1 block.
// 6. Use RUN_TESTS(suite) to run a test suite in the main function.
//
// Additional notes:
// - TEST_HOSTDEV(name) is a shortcut for TEST(name) and TEST_CUDA_KERNEL(name).

struct TestResult {
  int num_failures = 0;

  UM2_HOSTDEV void failure(char const * file, int line, char const * function, char const * message)
  {
    num_failures++;
    printf("Test failed in %s:%d\n", file, line);
    printf("  Function: %s\n", function);
    printf("  Message: %s", message); // Purposefully no newline
  }
};

inline void expect(bool a, TestResult * result, bool exit_on_failure, char const * const msg)
{
  if (!a) {
    result->failure(__FILE__, __LINE__, __FUNCTION__, msg);
    if (exit_on_failure) {
      exit(EXIT_FAILURE);
    }
  }
}

#define EXPECT_TRUE(a) \
  expect(a, result, exit_on_failure, "Expected " #a " to be true, but got false\n")

#define EXPECT_FALSE(a)                                                                            \
  expect(!(a), result, exit_on_failure, "Expected " #a " to be false, but got true\n")

#define EXPECT_EQ(a, b)                                                                            \
  expect((a) == (b), result, exit_on_failure,                                                      \
         "Expected " #a " == " #b ", but got " #a " != " #b "\n")

#define EXPECT_NE(a, b)                                                                            \
  expect((a) != (b), result, exit_on_failure,                                                      \
         "Expected " #a " != " #b ", but got " #a " == " #b "\n")

#define EXPECT_LT(a, b)                                                                            \
  expect((a) < (b), result, exit_on_failure,                                                       \
         "Expected " #a " < " #b ", but got " #a " >= " #b "\n")

#define EXPECT_LE(a, b)                                                                            \
  expect((a) <= (b), result, exit_on_failure,                                                      \
         "Expected " #a " <= " #b ", but got " #a " > " #b "\n")

#define EXPECT_GT(a, b)                                                                            \
  expect((a) > (b), result, exit_on_failure,                                                       \
         "Expected " #a " > " #b ", but got " #a " <= " #b "\n")

#define EXPECT_GE(a, b)                                                                            \
  expect((a) >= (b), result, exit_on_failure,                                                      \
         "Expected " #a " >= " #b ", but got " #a " < " #b "\n")

#define EXPECT_NEAR(a, b, eps)                                                                     \
  expect(std::abs((a) - (b)) < (eps), result, exit_on_failure,                                     \
         "Expected abs(" #a " - " #b ") < " #eps ", but got abs(" #a " - " #b ") = ");             \
  if (!(std::abs((a) - (b)) < (eps))) {                                                            \
    printf("%f\n", static_cast<double>(std::abs((a) - (b))));                                      \
  }

#define TEST_CASE(name) static void name(TestResult * const result, bool exit_on_failure)

#define TEST_SUITE(name) static void name(TestResult result, bool exit_on_failure)

#define TEST(name)                                                                                 \
  printf("Running test case '%s'\n", #name);                                                       \
  name(&result, exit_on_failure);                                                                  \
  printf("Test case '%s' finished with %d failures\n", #name, result.num_failures);                \
  if (result.num_failures > 0) {                                                                   \
    return;                                                                                        \
  }

#if UM2_ENABLE_CUDA
#  define MAKE_CUDA_KERNEL(host_test)                                                              \
    __global__ void host_test##_cuda_kernel(TestResult * const result,                             \
                                            bool * const exit_on_failure)                          \
    {                                                                                              \
      host_test(result, *exit_on_failure);                                                         \
    }

#  define __TEST_CUDA_KERNEL(host_test, blocks, threads)                                           \
    {                                                                                              \
      TestResult * device_result;                                                                  \
      cudaMalloc(&device_result, sizeof(TestResult));                                              \
      bool * device_exit_on_failure;                                                               \
      cudaMalloc(&device_exit_on_failure, sizeof(bool));                                           \
      cudaMemcpy(device_exit_on_failure, &exit_on_failure, sizeof(bool), cudaMemcpyHostToDevice);  \
      printf("Running CUDA test case '%s' with %d blocks and %d threads\n", #host_test, blocks,    \
             threads);                                                                             \
      host_test##_cuda_kernel<<<(blocks), (threads)>>>(device_result, device_exit_on_failure);     \
      cudaDeviceSynchronize();                                                                     \
      cudaError_t error = cudaGetLastError();                                                      \
      if (error != cudaSuccess) {                                                                  \
        printf("CUDA error: %s\n", cudaGetErrorString(error));                                     \
        return;                                                                                    \
      }                                                                                            \
      cudaMemcpy(&result, device_result, sizeof(TestResult), cudaMemcpyDeviceToHost);              \
      printf("CUDA test case '%s' finished with %d failures\n", #host_test, result.num_failures);  \
      if (result.num_failures > 0) {                                                               \
        return;                                                                                    \
      }                                                                                            \
    }

#  define TEST_CUDA_KERNEL_1_ARGS(host_test) __TEST_CUDA_KERNEL(host_test, 1, 1)

#  define TEST_CUDA_KERNEL_2_ARGS(host_test, threads) __TEST_CUDA_KERNEL(host_test, 1, threads)

#  define TEST_CUDA_KERNEL_3_ARGS(host_test, blocks, threads)                                      \
    __TEST_CUDA_KERNEL(host_test, blocks, threads)

#  define TEST_CUDA_KERNEL_GET_MACRO(_1, _2, _3, NAME, ...) NAME
#  define TEST_CUDA_KERNEL(...)                                                                    \
    TEST_CUDA_KERNEL_GET_MACRO(__VA_ARGS__, TEST_CUDA_KERNEL_3_ARGS, TEST_CUDA_KERNEL_2_ARGS,      \
                               TEST_CUDA_KERNEL_1_ARGS)                                            \
    (__VA_ARGS__)

#else
#  define MAKE_CUDA_KERNEL(host_test)
#  define TEST_CUDA_KERNEL(...)
#endif

#define RUN_THE_TESTS(suite, exit_on_failure)                                                      \
  {                                                                                                \
    TestResult result;                                                                             \
    printf("Running test suite '%s'\n", #suite);                                                   \
    suite(result, exit_on_failure);                                                                \
    printf("Test suite '%s' finished with %d failures\n", #suite, result.num_failures);            \
    if (result.num_failures > 0 && (exit_on_failure)) {                                            \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  }

#define RUN_TESTS_1_ARGS(suite) RUN_THE_TESTS(suite, true)

#define RUN_TESTS_2_ARGS(suite, exit_on_failure) RUN_THE_TESTS(suite, exit_on_failure)

#define RUN_TESTS_GET_MACRO(_1, _2, NAME, ...) NAME
#define RUN_TESTS(...)                                                                             \
  RUN_TESTS_GET_MACRO(__VA_ARGS__, RUN_TESTS_2_ARGS, RUN_TESTS_1_ARGS)(__VA_ARGS__)

#define TEST_HOSTDEV(name)                                                                         \
  TEST(name);                                                                                      \
  TEST_CUDA_KERNEL(name);
