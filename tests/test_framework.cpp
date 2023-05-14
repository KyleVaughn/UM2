#include "test_framework.hpp"

// cppcheck-suppress duplicateExpression
UM2_HOSTDEV TEST_CASE(expectTrue) { EXPECT_TRUE(1 == 1); }
MAKE_CUDA_KERNEL(expectTrue);

UM2_HOSTDEV TEST_CASE(expectFalse) { EXPECT_FALSE(1 == 2); }
MAKE_CUDA_KERNEL(expectFalse);

// cppcheck-suppress duplicateExpression
UM2_HOSTDEV TEST_CASE(expectEqual) { EXPECT_EQ(1, 1); }
MAKE_CUDA_KERNEL(expectEqual);

UM2_HOSTDEV TEST_CASE(expectNotEqual) { EXPECT_NE(1, 2); }
MAKE_CUDA_KERNEL(expectNotEqual);

UM2_HOSTDEV TEST_CASE(expectLess) { EXPECT_LT(1, 2); }
MAKE_CUDA_KERNEL(expectLess);

UM2_HOSTDEV TEST_CASE(expectLessEqual) { EXPECT_LE(1, 2); }
MAKE_CUDA_KERNEL(expectLessEqual);

UM2_HOSTDEV TEST_CASE(expectGreater) { EXPECT_GT(2, 1); }
MAKE_CUDA_KERNEL(expectGreater);

UM2_HOSTDEV TEST_CASE(expectGreaterEqual) { EXPECT_GE(2, 1); }
MAKE_CUDA_KERNEL(expectGreaterEqual);

UM2_HOSTDEV TEST_CASE(expectNear) { EXPECT_NEAR(1.0, 1.1, 0.2); }
MAKE_CUDA_KERNEL(expectNear);

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
