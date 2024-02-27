#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/utility/forward.hpp>
#include <um2/stdlib/utility/move.hpp>

#include <compare>

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
  HOSTDEV constexpr Pair(U1 && x, U2 && y) noexcept
      : first(um2::forward<U1>(x)),
        second(um2::forward<U2>(y))
  {
  }

  HOSTDEV constexpr auto operator=(Pair const &) noexcept -> Pair & = default;
  HOSTDEV constexpr auto operator=(Pair &&) noexcept -> Pair & = default;

  //============================================================================
  // Destructor
  //============================================================================

  HOSTDEV ~Pair() noexcept = default;

  //============================================================================
  // Relational operators
  //============================================================================

  // NOLINTNEXTLINE(modernize-use-nullptr)
  auto operator<=>(Pair const &) const = default;

};

//==============================================================================
// Non-member functions
//==============================================================================

//template <class T1, class T2>
//HOSTDEV constexpr auto
//operator==(Pair<T1, T2> const & x, Pair<T1, T2> const & y) noexcept -> bool
//{
//  return x.first == y.first && x.second == y.second;
//}

//template <class T1, class T2>
//HOSTDEV constexpr auto
//operator<(Pair<T1, T2> const & x, Pair<T1, T2> const & y) noexcept -> bool
//{
//  return x.first < y.first || (!(y.first < x.first) && x.second < y.second);
//}
//
//
//
//template <class T1, class T2>
//HOSTDEV constexpr auto
//operator!=(Pair<T1, T2> const & x, Pair<T1, T2> const & y) noexcept -> bool
//{
//  return !(x == y);
//}
//
//
//
//
//
//
//
//
//template <class T1, class T2>
//HOSTDEV constexpr auto
//operator>(Pair<T1, T2> const & x, Pair<T1, T2> const & y) noexcept -> bool
//{
//  return y < x;
//}
//
//template <class T1, class T2>
//HOSTDEV constexpr auto
//operator<=(Pair<T1, T2> const & x, Pair<T1, T2> const & y) noexcept -> bool
//{
//  return !(y < x);
//}
//
//template <class T1, class T2>
//HOSTDEV constexpr auto
//operator>=(Pair<T1, T2> const & x, Pair<T1, T2> const & y) noexcept -> bool
//{
//  return !(x < y);
//}

} // namespace um2
