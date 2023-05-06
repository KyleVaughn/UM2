#pragma once

#include <atomic> // std::atomic
#include <iostream> // std::cerr, std::cout, std::endl
#include <syncstream> // std::osyncstream

static std::atomic<int> failedChecks = 0;
static bool exitOnFailure = true;

#define EXPECT_EQ(a, b) \
    if (a != b) { \
      failedChecks++; \
      std::cerr << "Failed check: " << #a << " == " << #b << std::endl; \
      if (exitOnFailure) exit(1); \
    }

#define ASSERT_EQ(a, b) \
    if (a != b) { \
      failedChecks++; \
      std::cerr << "Failed check: " << #a << " == " << #b << std::endl; \
      exit(1); \
    }

#define TEST_CASE(name) \
    static void name()



//
//// A simple unit testing framework.
//// ---------------------------------------------------------------------------------
//// The main function in each test file should look like this:
////
//// int main() {
////    RUN_TEST_SUITE("Test Suite 1 Name", test_suite1);
////    RUN_TEST_SUITE("Test Suite 2 Name", test_suite2);
////    .... etc
////    return 0;
//// }
////
//// The RUN_TEST_SUITE function takes the display name of the test suite and a function
//// pointer to the test suite function.
////
//// A test suite is created using the TEST_SUITE(name) and END_TEST_SUITE macros.
//// A test suite us used to run multiple tests using the RUN_TEST and RUN_CUDA_TEST
//// macros. The test suite is a static void function that takes a string
//// (its display name) as an argument. The test suite also creates an error counter
//// and a pointer to the error counter. The error counter pointer is passed to each test as
//// an argument. The error counter is incremented if an assertion fails (more on this
//// later). Note the additional parentheses needed to pass the templated
//// function pointer to the macro in the example below.
////
//// template <typename T, typename U>
//// TEST_SUITE(test_suite1)
////   RUN_TEST("Test 1", (test1<T, U>));
////   RUN_TEST("Test 2", (test2<T, U>));
////   .... etc
////
////   RUN_CUDA_TEST("Test 1 cuda", (test1_cuda<T, U>));
//// END_TEST_SUITE
////
//// Tests are created using the TEST(name) and END_TEST macros. The ASSERT(bool, msg) and
//// ASSERT_APPROX(val1, val2, max_abs_diff, msg) macros are used to check if a condition is true.
//// Example:
////
//// template <typename T, typename U>
//// TEST(test1)
////   T const x = 1;
////   ASSERT(x == 1, "x == 1");
////   ASSERT(x != 2, "x != 2");
////   U const y = 2;
////   ASSERT_APPROX(y, 1.001, 0.1, "y is approx 1.001");
//// END_TEST
////
//// For CUDA tests, things are slightly more complicated. If a test has functions that
//// need to be run on the host and device, the UM2_HOSTDEV macro should be placed in front
//// of the function. Ex:
////
//// UM2_HOSTDEV TEST(test_int)
////   int const x = 1;
////   ASSERT(x == 1, "x == 1");
//// END_TEST
////
//// Then, in order to run the test on the device, a kernel which calls the test function
//// and a function to launch the kernel should be created.
//// The ADD_CUDA_TEST(host_test, cuda_test) macro can be used to create the kernel and launch
//// function using 1 block and 1 thread.
//// Example: ADD_CUDA_TEST(test_int, test_int_cuda). Then the test can be run using
//// RUN_CUDA_TEST("int_cuda", test_int_cuda) inside the test suite.
////
//// For templated tests, this process requires a little more work. The example below shows
//// the necessary steps to run a templated test on the device.
////
//// template <typename T>
//// UM2_HOSTDEV TEST(test_addition)
////   T const x = 1;
////   T const y = 2;
////   ASSERT(x + y == 3, "x + y == 3");
//// END_TEST
////
//// templated cuda test definitions must be wrapped in an #if UM2_HAS_CUDA block, otherwise
//// when CUDA is not enabled, the compiler will just see "template <typename T>" without a
//// function definition and throw an error.
//// #if UM2_HAS_CUDA
////   template <typename T>
////   ADD_TEMPLATED_CUDA_KERNEL(test_addition, test_addition_kernel, T);
////   template <typename T>
////   ADD_TEMPLATED_CUDA_TEST(test_addition_kernel, test_addition_cuda, T);
//// #endif
////
//// template <typename T>
//// TEST_SUITE(test_suite1)
////   RUN_TEST("addition", (test_addition<T>));
////   RUN_CUDA_TEST("addition_cuda", (test_addition_cuda<T>));
//// END_TEST_SUITE
////
//// int main() {
////  RUN_TEST_SUITE("suite_1", test_suite1<int>);
////  RUN_TEST_SUITE("suite_2", test_suite1<float>);
////  .... etc
////  return 0;
//// }
////
//
//// Log options
//// Log is configured at the start of each test suite
//// --------------------------------------------------
//#ifndef UM2_TEST_VERBOSITY
//#   define UM2_TEST_VERBOSITY 1
//#endif
//#if UM2_TEST_VERBOSITY == 0
//#   define UM2_TEST_LOG_VERBOSITY um2::LogVerbosity::error
//#elif UM2_TEST_VERBOSITY == 1
//#   define UM2_TEST_LOG_VERBOSITY um2::LogVerbosity::warn
//#elif UM2_TEST_VERBOSITY == 2
//#   define UM2_TEST_LOG_VERBOSITY um2::LogVerbosity::info
//#elif UM2_TEST_VERBOSITY == 3
//#   define UM2_TEST_LOG_VERBOSITY um2::LogVerbosity::debug
//#else
//#   error "Invalid UM2_TEST_VERBOSITY"
//#endif
//
//#ifndef UM2_TEST_EXIT_ON_ERROR
//#   define UM2_TEST_EXIT_ON_ERROR true
//#endif
//
//#ifndef UM2_TEST_COLOR_OUTPUT
//#   define UM2_TEST_COLOR_OUTPUT true
//#endif
//
//// If true, individual test results will be printed
//// If false, test suite results are still printed, but individual test results are not
//#ifndef UM2_SHOW_TEST_RESULTS
//#   define UM2_SHOW_TEST_RESULTS true
//#endif
//
//// The width of test messages.
//#define UM2_TEST_LOG_MSG_WIDTH 82
//
//std::string format_unit_test_message(
//        std::string const & msg,
//        bool const success,
//        um2::LogTimePoint const & start_time,
//        bool const skipped = false);
//
//// Test suite
//// ---------------------------------------------------------------------------------
//
//// Run the test suite. Arguments are the test suite's display name and a function
//// pointer to the test suite function.
//void RUN_TEST_SUITE(std::string const &, void (*test_suite)(std::string const &));
//
//// The test suite is a static void function that takes a string (its display name)
//// as an argument. The test suite should call the RUN_TEST and RUN_DEVICE_TEST macros
//// to run individual tests.
//#define TEST_SUITE(name)                                                            \
//        static void name(std::string const & suite_name) {                          \
//            int errors = 0;                                                         \
//            int * const errors_ptr = &errors;
//
//// At the end of the test suite, this macro prints the test suite's results:
//// The name of the display name of the suite, pass/fail/skip, and the time it took
//// to run the suite.
//#define END_TEST_SUITE                                                              \
//        bool const success = (errors == 0);                                         \
//        std::cerr << format_unit_test_message(                                      \
//                suite_name, success, um2::Log::get_start_time()) << std::endl;      \
//        if (!success && UM2_TEST_EXIT_ON_ERROR) {                                   \
//            exit(1);                                                                \
//        }                                                                           \
//    }
//
//// Test
//// ---------------------------------------------------------------------------------
//
//void _RUN_TEST(
//        std::string const & name,
//        void (*test)(int * const),
//        std::string const & suite_name,
//        int * const errors_ptr);
//
//#define RUN_TEST(name, ptr)                                                         \
//    _RUN_TEST(name, ptr, suite_name, errors_ptr)
//
//#define TEST(name)                                                                  \
//    static void name(int * const errors_ptr) {                                      \
//
//#define END_TEST                                                                    \
//    }
//
//// Assertions
//// ---------------------------------------------------------------------------------
//
//#if UM2_HAS_CUDA
//#   define ASSERT(bool, msg) if (!(bool)) { (*errors_ptr)++; }
//#else
//#   define ASSERT(bool, msg)                                                        \
//        if (!(bool)) {                                                              \
//            (*errors_ptr)++;                                                        \
//            um2::Log::error(std::string(__FILE__) + ":" +                           \
//                    std::to_string(__LINE__) + " '" + msg + "'");                   \
//        }
//#endif
//
//// Disable -Wconversion and -Wdouble-promotion warnings
//#define ASSERT_APPROX(a, b, tol, msg)                                               \
//    _Pragma("GCC diagnostic push")                                                  \
//    _Pragma("GCC diagnostic ignored \"-Wconversion\"")                              \
//    _Pragma("GCC diagnostic ignored \"-Wdouble-promotion\"")                        \
//    ASSERT(std::abs((a) - (b)) < (tol), msg)                                        \
//    _Pragma("GCC diagnostic pop")
//
//// CUDA
//// ---------------------------------------------------------------------------------
//#if UM2_HAS_CUDA
//
//void check_cuda_error(int * const errors_ptr, std::string * error_msg_ptr);
//void test_cuda_kernel(void (*kernel)(int * const),
//                      int * const errors_ptr);
//
//#define ADD_CUDA_TEST(host_test, cuda_test)                                         \
//    __global__ void                                                                 \
//    cuda_test##_kernel(int * const errors_ptr) {                                    \
//        host_test(errors_ptr);                                                      \
//    }                                                                               \
//    TEST(cuda_test)                                                                 \
//        test_cuda_kernel(cuda_test##_kernel, errors_ptr);                           \
//    END_TEST
//
//#define ADD_TEMPLATED_CUDA_KERNEL(host_test, kernel_name, ...)                      \
//    __global__ void                                                                 \
//    kernel_name(int * const errors_ptr) {                                           \
//        host_test<__VA_ARGS__>(errors_ptr);                                         \
//    }
//
//#define ADD_TEMPLATED_KERNEL_TEST(kernel_name, test_name, ...)                      \
//  TEST(test_name)                                                                   \
//        test_cuda_kernel((kernel_name<__VA_ARGS__>), errors_ptr);                   \
//  END_TEST
//
//#else
//#define ADD_CUDA_TEST(host_test, cuda_test)
//#define ADD_TEMPLATED_CUDA_KERNEL(host_test, kernel_name, ...)
//#define ADD_TEMPLATED_KERNEL_TEST(kernel_name, test_name, ...)
//#endif // UM2_HAS_CUDA
//
//void print_skipped_test(std::string suite_name, std::string test_name);
//#define SKIP_TEST(name) print_skipped_test(suite_name, name)
//
//#if UM2_HAS_CUDA
//#   define RUN_CUDA_TEST(name, ptr) RUN_TEST(name, ptr)
//#else
//#   define RUN_CUDA_TEST(name, ptr) SKIP_TEST(name)
//#endif
