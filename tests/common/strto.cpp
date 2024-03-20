#include <um2/common/strto.hpp>

#include "../test_macros.hpp"

HOSTDEV
TEST_CASE(strto_float)
{
  char const * const s = "123.456";
  char * end = nullptr;
  float const f = um2::strto<float>(s, &end);
  ASSERT_NEAR(f, 123.456F, 0.001F);
}

HOSTDEV
TEST_CASE(strto_double)
{
  char const * const s = "123.456";
  char * end = nullptr;
  double const d = um2::strto<double>(s, &end);
  ASSERT_NEAR(d, 123.456, 0.001);
}

HOSTDEV
TEST_CASE(strto_int32)
{
  char const * const s = "-123";
  char * end = nullptr;
  int32_t const i = um2::strto<int32_t>(s, &end);
  ASSERT(i == -123);
}

HOSTDEV
TEST_CASE(strto_uint32)
{
  char const * const s = "123";
  char * end = nullptr;
  uint32_t const u = um2::strto<uint32_t>(s, &end);
  ASSERT(u == 123);
}

HOSTDEV
TEST_CASE(strto_int64)
{
  char const * const s = "-123";
  char * end = nullptr;
  int64_t const i = um2::strto<int64_t>(s, &end);
  ASSERT(i == -123);
}

HOSTDEV
TEST_CASE(strto_uint64)
{
  char const * const s = "123";
  char * end = nullptr;
  uint64_t const u = um2::strto<uint64_t>(s, &end);
  ASSERT(u == 123);
}

TEST_SUITE(strto)
{
  TEST_HOSTDEV(strto_float);
  TEST_HOSTDEV(strto_double);
  TEST_HOSTDEV(strto_int32);
  TEST_HOSTDEV(strto_uint32);
  TEST_HOSTDEV(strto_int64);
  TEST_HOSTDEV(strto_uint64);
}

auto
main() -> int
{
  RUN_SUITE(strto);
  return 0;
}
