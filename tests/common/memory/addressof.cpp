#include <um2/common/memory/addressof.hpp>

#include "../../test_macros.hpp"


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
TEST_CASE(addressof)
{
  // NOLINTBEGIN(misc-static-assert)
  {
    int i = 0;
    double d = 0;
    ASSERT(um2::addressof(i) == &i);
    ASSERT(um2::addressof(d) == &d);

    A * tp = new A;
    const A * ctp = tp;
    ASSERT(um2::addressof(*tp) == tp);
    ASSERT(um2::addressof(*ctp) == ctp);
    delete tp;
  }
  {
    union {
      Nothing n;
      int i;
    };
    ASSERT(um2::addressof(n) == static_cast<void *>(um2::addressof(n)));
  }
  // NOLINTEND(misc-static-assert)
}
MAKE_CUDA_KERNEL(addressof);

TEST_SUITE(addressof) { TEST_HOSTDEV(addressof); }

auto
main() -> int
{
  RUN_SUITE(addressof);
  return 0;
}
