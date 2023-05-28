#include "../test_framework.hpp"
#include <um2/common/bit_ceil.hpp>

template <typename T>
UM2_HOSTDEV TEST_CASE(zero_through_5)
{
  printf("bit_ceil(%d) = %d\n", 0, um2::bit_ceil(0));
  EXPECT_EQ(um2::bit_ceil(0), 1);
  EXPECT_EQ(um2::bit_ceil(1), 1);
  EXPECT_EQ(um2::bit_ceil(2), 2);
  EXPECT_EQ(um2::bit_ceil(3), 4);
  EXPECT_EQ(um2::bit_ceil(4), 4);
  EXPECT_EQ(um2::bit_ceil(5), 8);
}

#if UM2_ENABLE_CUDA
template <typename T>
MAKE_CUDA_KERNEL(zero_through_5, T);
#endif

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
