#include "../test_framework.hpp"
#include <um2/common/sto.hpp>

TEST_CASE(sto_int16)
{
  int16_t i = 0;
  i = um2::sto<int16_t>("123");
  EXPECT_EQ(i, 123);
}

TEST_CASE(sto_int32)
{
  int32_t i = 0;
  i = um2::sto<int32_t>("123");
  EXPECT_EQ(i, 123);
}

TEST_CASE(sto_int64)
{
  int64_t i = 0;
  i = um2::sto<int64_t>("123");
  EXPECT_EQ(i, 123);
}

TEST_CASE(sto_uint16)
{
  uint16_t i = 0;
  i = um2::sto<uint16_t>("123");
  EXPECT_EQ(i, 123);
}

TEST_CASE(sto_uint32)
{
  uint32_t i = 0;
  i = um2::sto<uint32_t>("123");
  EXPECT_EQ(i, 123);
}

TEST_CASE(sto_uint64)
{
  uint64_t i = 0;
  i = um2::sto<uint64_t>("123");
  EXPECT_EQ(i, 123);
}

// Ignore float equality warnings, since we're testing true equality
TEST_CASE(sto_float)
{
  float f = 0;
  f = um2::sto<float>("123.456");
  EXPECT_NEAR(f, 123.456F, 1e-6F);
}

TEST_CASE(sto_double)
{
  double d = 0;
  d = um2::sto<double>("123.456");
  EXPECT_NEAR(d, 123.456, 1e-6);
}

TEST_SUITE(sto)
{
  TEST(sto_int16);
  TEST(sto_int32);
  TEST(sto_int64);
  TEST(sto_uint16);
  TEST(sto_uint32);
  TEST(sto_uint64);
  TEST(sto_float);
  TEST(sto_double);
}

auto main() -> int
{
  RUN_TESTS(sto);
  return 0;
}
