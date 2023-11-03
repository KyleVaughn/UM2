#include <um2/stdlib/ato.hpp>

#include "../test_macros.hpp"

TEST_CASE(ato_int16)
{
  int16_t i = 0;
  i = um2::ato<int16_t>("123");
  ASSERT(i == 123);
}

TEST_CASE(ato_int32)
{
  int32_t i = 0;
  i = um2::ato<int32_t>("123");
  ASSERT(i == 123);
}

TEST_CASE(ato_int64)
{
  int64_t i = 0;
  i = um2::ato<int64_t>("123");
  ASSERT(i == 123);
}

TEST_CASE(ato_uint16)
{
  uint16_t i = 0;
  i = um2::ato<uint16_t>("123");
  ASSERT(i == 123);
}

TEST_CASE(ato_uint32)
{
  uint32_t i = 0;
  i = um2::ato<uint32_t>("123");
  ASSERT(i == 123);
}

TEST_CASE(ato_uint64)
{
  uint64_t i = 0;
  i = um2::ato<uint64_t>("123");
  ASSERT(i == 123);
}

#ifndef __clang__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif

TEST_CASE(ato_float)
{
  float f = 0;
  f = um2::ato<float>("123.456");
  float const f_expected = 123.456F;
  ASSERT_NEAR(f, f_expected, 1e-6F);
}

TEST_CASE(ato_double)
{
  double d = 0;
  d = um2::ato<double>("123.456");
  ASSERT_NEAR(d, 123.456, 1e-6);
}

TEST_SUITE(ato)
{
  TEST(ato_int16);
  TEST(ato_int32);
  TEST(ato_int64);
  TEST(ato_uint16);
  TEST(ato_uint32);
  TEST(ato_uint64);
  TEST(ato_float);
  TEST(ato_double);
}

auto
main() -> int
{
  RUN_SUITE(ato);
  return 0;
}

#ifndef __clang__
#  pragma GCC diagnostic pop
#endif
