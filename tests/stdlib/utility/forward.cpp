#include <um2/stdlib/utility/forward.hpp>

#include <utility> // std::move, since we um2::move is not tested yet at this step

#include "../../test_macros.hpp"

struct A
{
  int l;
  int r;
};

HOSTDEV
auto
value(A& a) -> int
{
  return a.l;
}

HOSTDEV
auto
value(A&& a) -> int
{
  return a.r;
}

template <typename T>
HOSTDEV
auto
pass(T&& a) -> int
{
  return value(um2::forward<T>(a));
}

HOSTDEV
TEST_CASE(test_forward)
{
  A a{1, 2};
  auto l = pass(a);
  // NOLINTNEXTLINE(performance-move-const-arg)
  auto r = pass(std::move(a));
  ASSERT(l == 1);
  ASSERT(r == 2);
}
MAKE_CUDA_KERNEL(test_forward);

TEST_SUITE(forward) { TEST_HOSTDEV(test_forward); }

auto
main() -> int
{
  RUN_SUITE(forward);
  return 0;
}
