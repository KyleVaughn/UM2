#pragma once

#include <um2/config.hpp>

#include <type_traits>

//  forward
//  move
//  swap
//  Pair

namespace um2
{

//==============================================================================
// forward
//==============================================================================
//
// https://en.cppreference.com/w/cpp/utility/forward

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
forward(std::remove_reference_t<T> & t) noexcept -> T &&
{
  // Forwards lvalues as either lvalues or as rvalues, depending on T.
  return static_cast<T &&>(t);
}

template <class T>
HOSTDEV [[nodiscard]] constexpr auto
forward(std::remove_reference_t<T> && t) noexcept -> T &&
{
  // Forwards rvalues as rvalues and prohibits forwarding of lvalues.
  static_assert(!std::is_lvalue_reference_v<T>,
                "Can not forward an rvalue as an lvalue.");
  return static_cast<T &&>(t);
}

//==============================================================================
// move
//==============================================================================
//
// https://en.cppreference.com/w/cpp/utility/move

template <class T>
HOSTDEV constexpr auto
move(T && t) noexcept -> std::remove_reference_t<T> &&
{
  return static_cast<std::remove_reference_t<T> &&>(t);
}

//==============================================================================
// swap
//==============================================================================
//
// https://en.cppreference.com/w/cpp/utility/swap

template <class T>
  requires(std::is_trivially_move_constructible_v<T> &&
           std::is_trivially_move_assignable_v<T>)
HOSTDEV constexpr void swap(T & a, T & b) noexcept
{
  T tmp = um2::move(a);
  a = um2::move(b);
  b = um2::move(tmp);
}

//==============================================================================
// Pair
//==============================================================================

// NOLINTBEGIN(readability-identifier-naming) // match std::pair
template <class T1, class T2>
struct Pair {
  using first_type = T1;
  using second_type = T2;

  T1 first;
  T2 second;

  //============================================================================
  // Constructors
  //============================================================================

  HOSTDEV
  constexpr Pair(T1 x, T2 y)
      : first(x),
        second(y)
  {
  }

  template <class U1, class U2>
  HOSTDEV constexpr Pair(U1 && x, U2 && y)
      : first(um2::forward<U1>(x)),
        second(um2::forward<U2>(y))
  {
  }

};

template <class T1, class T2>
HOSTDEV constexpr auto
operator==(Pair<T1, T2> const & x, Pair<T1, T2> const & y) -> bool
{
  return x.first == y.first && x.second == y.second;
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator!=(Pair<T1, T2> const & x, Pair<T1, T2> const & y) -> bool
{
  return !(x == y);
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator<(Pair<T1, T2> const & x, Pair<T1, T2> const & y) -> bool
{
  return x.first < y.first || (!(y.first < x.first) && x.second < y.second);
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator>(Pair<T1, T2> const & x, Pair<T1, T2> const & y) -> bool
{
  return y < x;
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator<=(Pair<T1, T2> const & x, Pair<T1, T2> const & y) -> bool
{
  return !(y < x);
}

template <class T1, class T2>
HOSTDEV constexpr auto
operator>=(Pair<T1, T2> const & x, Pair<T1, T2> const & y) -> bool
{
  return !(x < y);
}

template <class T1, class T2>
HOSTDEV constexpr auto
make_pair(T1 && x, T2 && y) -> Pair<T1, T2>
{
  return Pair<T1, T2>(um2::forward<T1>(x), um2::forward<T2>(y));
}

template <class T1, class T2>
HOSTDEV constexpr auto
make_pair(T1 const & x, T2 const & y) -> Pair<T1, T2>
{
  return Pair<T1, T2>(x, y);
}

// NOLINTEND(readability-identifier-naming)

} // namespace um2
