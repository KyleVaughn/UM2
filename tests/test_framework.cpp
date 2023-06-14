#include "test_framework.hpp"

#ifdef __clang__
// Ignore unreachable code warnings for the tests, since
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wunreachable-code"
#endif
// cppcheck-suppress duplicateExpression
HOSTDEV TEST_CASE(expectTrue) { EXPECT_TRUE(1 == 1); }
MAKE_CUDA_KERNEL(expectTrue);

HOSTDEV TEST_CASE(expectFalse) { EXPECT_FALSE(1 == 2); }
MAKE_CUDA_KERNEL(expectFalse);

// cppcheck-suppress duplicateExpression
HOSTDEV TEST_CASE(expectEqual) { EXPECT_EQ(1, 1); }
MAKE_CUDA_KERNEL(expectEqual);

HOSTDEV TEST_CASE(expectNotEqual) { EXPECT_NE(1, 2); }
MAKE_CUDA_KERNEL(expectNotEqual);

HOSTDEV TEST_CASE(expectLess) { EXPECT_LT(1, 2); }
MAKE_CUDA_KERNEL(expectLess);

HOSTDEV TEST_CASE(expectLessEqual) { EXPECT_LE(1, 2); }
MAKE_CUDA_KERNEL(expectLessEqual);

HOSTDEV TEST_CASE(expectGreater) { EXPECT_GT(2, 1); }
MAKE_CUDA_KERNEL(expectGreater);

HOSTDEV TEST_CASE(expectGreaterEqual) { EXPECT_GE(2, 1); }
MAKE_CUDA_KERNEL(expectGreaterEqual);

HOSTDEV TEST_CASE(expectNear) { EXPECT_NEAR(1.0, 1.1, 0.2); }
MAKE_CUDA_KERNEL(expectNear);
#ifdef __clang__
#  pragma clang diagnostic pop
#endif

TEST_SUITE(testFramework)
{
  TEST_HOSTDEV(expectTrue);
  TEST_HOSTDEV(expectFalse);
  TEST_HOSTDEV(expectEqual);
  TEST_HOSTDEV(expectNotEqual);
  TEST_HOSTDEV(expectLess);
  TEST_HOSTDEV(expectLessEqual);
  TEST_HOSTDEV(expectGreater);
  TEST_HOSTDEV(expectGreaterEqual);
  TEST_HOSTDEV(expectNear);
}

auto main() -> int
{
  RUN_TESTS(testFramework);
  return 0;
}
