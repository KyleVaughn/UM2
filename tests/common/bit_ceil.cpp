#include "../test_framework.hpp"
#include <um2/common/bit_ceil.hpp>

template <typename T>
TEST_CASE(zero_through_5)
{
  EXPECT_EQ(um2::bit_ceil<T>(0), 1);
  EXPECT_EQ(um2::bit_ceil<T>(1), 1);
  EXPECT_EQ(um2::bit_ceil<T>(2), 2);
  EXPECT_EQ(um2::bit_ceil<T>(3), 4);
  EXPECT_EQ(um2::bit_ceil<T>(4), 4);
  EXPECT_EQ(um2::bit_ceil<T>(5), 8);
}

template <typename T>
TEST_SUITE(bit_ceil)
{
  TEST_HOSTDEV(zero_through_5, 1, 1, T);
}

auto main() -> int
{
  RUN_TESTS(bit_ceil<uint32_t>);
  RUN_TESTS(bit_ceil<uint64_t>);
  RUN_TESTS(bit_ceil<int32_t>);
  RUN_TESTS(bit_ceil<int64_t>);
  return 0;
}
