#include <um2/config.hpp>
#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/pair.hpp>

#include "../../test_macros.hpp"

#include <cstdint>

template <class T, class U>
HOSTDEV
TEST_CASE(test_pair)
{
  um2::Pair<T, U> p{static_cast<T>(1), static_cast<U>(2)};

  // Copy constructor
  um2::Pair<T, U> const p2{p};
  ASSERT(p2.first == 1);
  ASSERT(p2.second == 2);

  // Move constructor
  um2::Pair<T, U> const p3{um2::move(p)};
  ASSERT(p3.first == 1);
  ASSERT(p3.second == 2);

  // Explicit constructor
  um2::Pair<T, U> p4(static_cast<T>(1), static_cast<U>(2));
  ASSERT(p4.first == 1);
  ASSERT(p4.second == 2);

  // Copy assignment
  um2::Pair<T, U> p5 = p4;
  ASSERT(p5.first == 1);
  ASSERT(p5.second == 2);

  // Move assignment
  um2::Pair<T, U> const p6 = um2::move(p5);
  ASSERT(p6.first == 1);
  ASSERT(p6.second == 2);

  // Relational operators
  ASSERT(p2 == p3);
  ASSERT(p2 >= p3);
  ASSERT(p2 <= p3);
  p4.first = 3;
  ASSERT(p2 != p4);
  ASSERT(p2 < p4);
  ASSERT(p2 <= p4);
  ASSERT(p4 > p2);
  ASSERT(p4 >= p2);
  p4.first = 1;
  p4.second = 3;
  ASSERT(p2 < p4);
  ASSERT(p4 > p2);
}

template <class T, class U>
HOSTDEV
TEST_CASE(test_pair_constexpr)
{
  um2::Pair<T, U> constexpr p{static_cast<T>(1), static_cast<U>(2)};

  // Copy constructor
  um2::Pair<T, U> constexpr p2{p};
  static_assert(p2.first == 1);
  static_assert(p2.second == 2);

  // Explicit constructor
  um2::Pair<T, U> constexpr p3(static_cast<T>(1), static_cast<U>(3));
  static_assert(p3.first == 1);
  static_assert(p3.second == 3);

  // Copy assignment
  um2::Pair<T, U> constexpr p4 = p3;
  static_assert(p4.first == 1);
  static_assert(p4.second == 3);

  // Relational operators
  static_assert(p == p2);
  static_assert(p2 <= p3);
  static_assert(p3 >= p2);
  static_assert(p2 != p4);
  static_assert(p2 < p4);
  static_assert(p4 > p2);
}

#if UM2_USE_CUDA
template <class T, class U>
MAKE_CUDA_KERNEL(test_pair, T, U);

template <class T, class U>
MAKE_CUDA_KERNEL(test_pair_constexpr, T, U);
#endif

template <class T, class U>
TEST_SUITE(pair)
{
  TEST_HOSTDEV(test_pair, T, U);
  TEST_HOSTDEV(test_pair_constexpr, T, U);
}

auto
main() -> int
{
  RUN_SUITE((pair<int32_t, int32_t>));
  RUN_SUITE((pair<int64_t, uint32_t>));
  return 0;
}
