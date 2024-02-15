#include <um2/stdlib/memory/addressof.hpp>

#include "../../test_macros.hpp"

// NOLINTBEGIN justification: Just simple test code
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
    static char c = 'a';
    return c;
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

  A * tp = new A;
  A const * ctp = tp;
  ASSERT(um2::addressof(*tp) == tp);
  ASSERT(um2::addressof(*ctp) == ctp);
  delete tp;
  union {
    Nothing n;
    int j;
  };
// Clang can do this as a static assert, gcc cannot
#ifdef __clang__
    static_assert(um2::addressof(n) == static_cast<void *>(um2::addressof(n)));
#else
    ASSERT(um2::addressof(n) == static_cast<void *>(um2::addressof(n)));
#endif
}

MAKE_CUDA_KERNEL(test_addressof);

TEST_SUITE(addressof) { TEST_HOSTDEV(test_addressof); }

auto
main() -> int
{
  RUN_SUITE(addressof);
  return 0;
}
