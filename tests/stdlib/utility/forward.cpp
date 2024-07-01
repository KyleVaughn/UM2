#include <um2/config.hpp>
#include <um2/stdlib/utility/forward.hpp>

#include <utility> // std::move, since we um2::move is not tested yet at this step

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
// NOLINTNEXTLINE(*param-not-moved) OK
value(A && a) -> int
{
  return a.r;
}

HOSTDEV PURE constexpr auto
// NOLINTNEXTLINE(*param-not-moved) OK
value(A const && a) -> int
{
  return a.r;
}

template <typename T>
HOSTDEV constexpr auto
// NOLINTNEXTLINE(*missing-std-forward) OK
pass(T && a) -> int
{
  return value(um2::forward<T>(a));
}

HOSTDEV
TEST_CASE(test_forward)
{
  A a{1, 2};
  auto l = pass(a);
  // NOLINTNEXTLINE(performance-move-const-arg) OK
  auto r = pass(std::move(a));
  ASSERT(l == 1);
  ASSERT(r == 2);
}
MAKE_CUDA_KERNEL(test_forward);

HOSTDEV
TEST_CASE(test_forward_constexpr)
{
  A constexpr a{1, 2};
  auto constexpr l = pass(a);
  // NOLINTNEXTLINE(performance-move-const-arg) OK
  auto constexpr r = pass(std::move(a));
  static_assert(l == 1);
  static_assert(r == 2);
}
MAKE_CUDA_KERNEL(test_forward_constexpr);

TEST_SUITE(forward)
{
  TEST_HOSTDEV(test_forward);
  TEST_HOSTDEV(test_forward_constexpr);
}

auto
main() -> int
{
  RUN_SUITE(forward);
  return 0;
}
