#include <um2/config.hpp>
#include <um2/stdlib/memory/addressof.hpp>

#include "../../test_macros.hpp"

// addressof(x) must return the same value as &x, even if operator& is overloaded
struct A {
  int i;

  // NOLINTNEXTLINE(google-runtime-operator) we are testing operator&
  auto
  operator&() const -> int
  {
    return 42;
  }
};

HOSTDEV
TEST_CASE(test_addressof)
{
  // Basic tests
  int i = 0;
  double d = 0;
  static_assert(um2::addressof(i) == &i);
  static_assert(um2::addressof(d) == &d);

  // Overloaded operator&
  A * tp = new A;
  A const * ctp = tp;
  ASSERT(um2::addressof(*tp) == tp);
  ASSERT(um2::addressof(*ctp) == ctp);
  delete tp;

  // Constexpr tests
  constexpr int ci = 0;
  constexpr double cd = 0;
  static_assert(um2::addressof(ci) == &ci);
  static_assert(um2::addressof(cd) == &cd);
}
MAKE_CUDA_KERNEL(test_addressof);

TEST_SUITE(addressof) { TEST_HOSTDEV(test_addressof); }

auto
main() -> int
{
  RUN_SUITE(addressof);
  return 0;
}
