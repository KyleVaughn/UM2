#include "test_framework.hpp"

TEST_CASE(expect_eq)
{
  EXPECT_EQ(1, 1);
  EXPECT_EQ(1, 2);
}

int main()
{
  expect_eq();
  return 0;
}
