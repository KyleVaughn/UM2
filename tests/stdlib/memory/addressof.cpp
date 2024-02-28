#include <um2/stdlib/memory/addressof.hpp>

#include "../../test_macros.hpp"

// NOLINTBEGIN justification: Just simple test code
struct A {
  void
  operator&() const
  {
  }
};
// NOLINTEND


HOSTDEV
TEST_CASE(test_addressof)
{
  int i = 0;
  double d = 0;
  static_assert(um2::addressof(i) == &i);
  static_assert(um2::addressof(d) == &d);

  constexpr int ci = 0;
  constexpr double cd = 0;
  static_assert(um2::addressof(ci) == &ci);
  static_assert(um2::addressof(cd) == &cd);

  A * tp = new A;
  A const * ctp = tp;
  ASSERT(um2::addressof(*tp) == tp);
  ASSERT(um2::addressof(*ctp) == ctp);
  delete tp;
}

MAKE_CUDA_KERNEL(test_addressof);

TEST_SUITE(addressof) { TEST_HOSTDEV(test_addressof); }

auto
main() -> int
{
  RUN_SUITE(addressof);
  return 0;
}
