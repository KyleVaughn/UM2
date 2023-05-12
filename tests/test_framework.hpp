#pragma once

#include <cmath> // std::abs
#include <stdio.h> // printf

// A very simple GPU-compatible test framework.

#define PRINT_FAILURE_LOCATION() \
    fprintf(stderr, "Failed check in %s: %s:%d\n", __FUNCTION__, __FILE__, __LINE__);

#define EXPECT_EQ(a, b) \
    if (a != b) { \
      PRINT_FAILURE_LOCATION(); \
      fprintf(stderr, "  Expected %s == %s, but got %s != %s\n", #a, #b, #a, #b); \
      if (exit_on_failure) return; \
    }

#define EXPECT_APPROX_EQ(a, b, eps) \
    if (std::abs(a - b) > eps) { \
      PRINT_FAILURE_LOCATION(); \
      fprintf(stderr, "  Expected abs(%s, %s) < %s, but got abs(%s, %s) >= %s\n", \
        #a, #b, #eps, #a, #b, #eps); \
      if (exit_on_failure) return; \
    }
//
//#define ASSERT_EQ(a, b) \
//    if (a != b) { \
//      (*num_failed_checks)++; \
//      printf("Expected %s == %s, but got %s != %s\n", #a, #b, #a, #b); \
//      return; \
//    }

#define TEST_CASE(name) \
    static void name(bool exit_on_failure)

#define TEST_SUITE(name) \
    static void name(bool exit_on_failure)

#define TEST(name) name(exit_on_failure) 

#define RUN_TEST_SUITE_1_ARGS(suite) suite(true)
#define RUN_TEST_SUITE_2_ARGS(suite, exit_on_failure) suite(exit_on_failure)
#define RUN_TEST_SUITE_GET_MACRO(_1, _2, NAME, ...) NAME
#define RUN_TEST_SUITE(...) \
          RUN_TEST_SUITE_GET_MACRO(__VA_ARGS__, RUN_TEST_SUITE_2_ARGS, RUN_TEST_SUITE_1_ARGS) \
            (__VA_ARGS__)
