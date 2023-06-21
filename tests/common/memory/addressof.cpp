#include "../../test_macros.hpp"
#include <um2/common/memory/addressof.hpp>

// NOLINTBEGIN
struct A {
  void
  operator&() const
  {
  }
};

struct Nothing {
  explicit
  operator char &()
  {
    static char c;
    return c;
  }
};
// NOLINTEND

// ------------------------------------------------------------
// addressof
// ------------------------------------------------------------

HOSTDEV
TEST_CASE(test_addressof)
{
  // NOLINTBEGIN(misc-static-assert)
  {
    int i = 0;
    double d = 0;
    assert(um2::addressof(i) == &i);
    assert(um2::addressof(d) == &d);

    A * tp = new A;
    const A * ctp = tp;
    assert(um2::addressof(*tp) == tp);
    assert(um2::addressof(*ctp) == ctp);
    delete tp;
  }
  {
    union {
      Nothing n;
      int i;
    };
    assert(um2::addressof(n) == static_cast<void *>(um2::addressof(n)));
  }
  // NOLINTEND(misc-static-assert)
}
MAKE_CUDA_KERNEL(test_addressof);

TEST_SUITE(addressof) { TEST_HOSTDEV(test_addressof); }

auto
main() -> int
{
  RUN_TESTS(addressof);
  return 0;
}
