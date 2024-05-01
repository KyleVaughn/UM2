#include <um2/stdlib/utility/forward.hpp>
#include <um2/stdlib/utility/move.hpp>

#include "../../test_macros.hpp"

struct A {
  int l;
  int r;
};

HOSTDEV PURE constexpr auto
value(A & a) -> int
{
  return a.l;
}

HOSTDEV PURE constexpr auto
value(A const & a) -> int
{
  return a.l;
}

HOSTDEV PURE constexpr auto
value(A && a) -> int
{
  return a.r;
}

HOSTDEV PURE constexpr auto
value(A const && a) -> int
{
  return a.r;
}

template <typename T>
HOSTDEV constexpr auto
pass(T && a) -> int
{
  return value(um2::forward<T>(a));
}

HOSTDEV
TEST_CASE(test_move)
{
  A a{1, 2};
  auto l = pass(a);
  // NOLINTNEXTLINE(performance-move-const-arg) OK
  auto r = pass(um2::move(a));
  ASSERT(l == 1);
  ASSERT(r == 2);
}
MAKE_CUDA_KERNEL(test_move)

HOSTDEV
TEST_CASE(test_move_constexpr)
{
  A constexpr a{1, 2};
  auto constexpr l = pass(a);
  // NOLINTNEXTLINE(performance-move-const-arg) OK
  auto constexpr r = pass(um2::move(a));
  static_assert(l == 1);
  static_assert(r == 2);
}
MAKE_CUDA_KERNEL(test_move_constexpr)

TEST_SUITE(move) { TEST_HOSTDEV(test_move); }

auto
main() -> int
{
  RUN_SUITE(move);
  return 0;
}
