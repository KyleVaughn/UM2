#include "test_framework.hpp"

TEST_CASE(expectEQ)
{
  EXPECT_EQ(1, 1)
}

TEST_CASE(falseExpectEQ)
{
  EXPECT_EQ(1, 2)
}

TEST_CASE(expectNE)

TEST_SUITE(testFramework)
{
  TEST(expectEQ);
  bool tmp_exit_on_failure = exit_on_failure;
  exit_on_failure = false;
  TEST(falseExpectEQ);
  exit_on_failure = tmp_exit_on_failure;

}

auto main() -> int
{
  RUN_TEST_SUITE(testFramework);
  return 0;
}
