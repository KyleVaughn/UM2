#include <um2/stdlib/math/copysign.hpp>

#include "../../test_macros.hpp"

// NOLINTBEGIN(cert-dcl03-c,misc-static-assert)

HOSTDEV
TEST_CASE(copysign_float)
{
  ASSERT_NEAR(um2::copysign(1.0F, 2.0F), 1.0F, 1e-6F);
  ASSERT_NEAR(um2::copysign(1.0F, -2.0F), -1.0F, 1e-6F);
}
MAKE_CUDA_KERNEL(copysign_float);

HOSTDEV
TEST_CASE(copysign_double)
{
  ASSERT_NEAR(um2::copysign(1.0, 2.0), 1.0, 1e-6);
  ASSERT_NEAR(um2::copysign(1.0, -2.0), -1.0, 1e-6);
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)

TEST_SUITE(copysign)
{
  TEST_HOSTDEV(copysign_float);
  TEST_HOSTDEV(copysign_double);
}

auto
main() -> int
{
  RUN_SUITE(copysign);
  return 0;
}
