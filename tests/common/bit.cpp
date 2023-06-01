#include "../test_framework.hpp"
#include <um2/common/bit.hpp>

UM2_HOSTDEV TEST_CASE(bit_cast)
{
  uint8_t u8 = 255;
  int8_t i8 = -1;
  EXPECT_EQ(um2::bit_cast<int8_t>(u8), i8);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(bit_width)
{
  EXPECT_EQ(um2::bit_width(static_cast<T>(0)), 0);
  EXPECT_EQ(um2::bit_width(static_cast<T>(1)), 1);
  EXPECT_EQ(um2::bit_width(static_cast<T>(2)), 2);
  EXPECT_EQ(um2::bit_width(static_cast<T>(3)), 2);
  EXPECT_EQ(um2::bit_width(static_cast<T>(4)), 3);
  EXPECT_EQ(um2::bit_width(static_cast<T>(5)), 3);
  EXPECT_EQ(um2::bit_width(static_cast<T>(6)), 3);
  EXPECT_EQ(um2::bit_width(static_cast<T>(7)), 3);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(bit_ceil)
{
  EXPECT_EQ(um2::bit_ceil(static_cast<T>(0)), 1);
  EXPECT_EQ(um2::bit_ceil(static_cast<T>(1)), 1);
  EXPECT_EQ(um2::bit_ceil(static_cast<T>(2)), 2);
  EXPECT_EQ(um2::bit_ceil(static_cast<T>(3)), 4);
  EXPECT_EQ(um2::bit_ceil(static_cast<T>(4)), 4);
  EXPECT_EQ(um2::bit_ceil(static_cast<T>(5)), 8);
}

#if UM2_ENABLE_CUDA
MAKE_CUDA_KERNEL(bit_cast);

template <typename T>
MAKE_CUDA_KERNEL(bit_width, T);

template <typename T>
MAKE_CUDA_KERNEL(bit_ceil, T);
#endif

template <typename T>
TEST_SUITE(bit)
{
  TEST_HOSTDEV(bit_cast, 1, 1);
  TEST_HOSTDEV(bit_width, 1, 1, T);
  TEST_HOSTDEV(bit_ceil, 1, 1, T);
}

auto main() -> int
{
  RUN_TESTS(bit<uint32_t>);
  RUN_TESTS(bit<uint64_t>);
  RUN_TESTS(bit<int32_t>);
  RUN_TESTS(bit<int64_t>);
  return 0;
}
