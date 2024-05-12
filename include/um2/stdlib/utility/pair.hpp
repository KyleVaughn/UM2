#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/utility/forward.hpp>
#include <um2/stdlib/utility/move.hpp>

namespace um2
{

template <class T1, class T2>
struct Pair {

  T1 first;
  T2 second;

  //============================================================================
  // Constructors and assignment operators
  //============================================================================

  HOSTDEV constexpr Pair() noexcept = default;
  HOSTDEV constexpr Pair(Pair const &) noexcept = default;
  HOSTDEV constexpr Pair(Pair &&) noexcept = default;
  HOSTDEV
  constexpr Pair(T1 x, T2 y) noexcept
      : first(um2::move(x)),
        second(um2::move(y))
  {
  }

  template <class U1, class U2>
  // NOLINTNEXTLINE(*missing-std-forward) OK
  HOSTDEV constexpr Pair(U1 && x, U2 && y) noexcept
      : first(um2::forward<U1>(x)),
        second(um2::forward<U2>(y))
  {
  }

  HOSTDEV constexpr auto
  operator=(Pair const &) noexcept -> Pair & = default;
  HOSTDEV constexpr auto
  operator=(Pair &&) noexcept -> Pair & = default;

  //============================================================================
  // Destructor
  //============================================================================

  HOSTDEV ~Pair() noexcept = default;
};

//==============================================================================
// Non-member functions
//==============================================================================

template <class T1, class T2>
HOSTDEV constexpr auto
operator==(Pair<T1, T2> const & lhs, Pair<T1, T2> const & rhs) noexcept -> bool
{
  return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator<(Pair<T1, T2> const & lhs, Pair<T1, T2> const & rhs) noexcept -> bool
{
  return lhs.first < rhs.first || (!(rhs.first < lhs.first) && lhs.second < rhs.second);
}

// https://en.cppreference.com/w/cpp/language/operators#Comparison_operators
// lhs != rhs is equivalent to !(lhs == rhs)
// lhs > rhs is equivalent to rhs < lhs
// lhs <= rhs is equivalent to !(rhs < lhs)
// lhs >= rhs is equivalent to !(lhs < rhs)

template <class T1, class T2>
HOSTDEV constexpr auto
operator!=(Pair<T1, T2> const & lhs, Pair<T1, T2> const & rhs) noexcept -> bool
{
  return !(lhs == rhs);
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator>(Pair<T1, T2> const & lhs, Pair<T1, T2> const & rhs) noexcept -> bool
{
  return rhs < lhs;
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator<=(Pair<T1, T2> const & lhs, Pair<T1, T2> const & rhs) noexcept -> bool
{
  return !(rhs < lhs);
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator>=(Pair<T1, T2> const & lhs, Pair<T1, T2> const & rhs) noexcept -> bool
{
  return !(lhs < rhs);
}

} // namespace um2
