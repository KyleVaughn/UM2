#pragma once

#include <um2/config.hpp>

#include <algorithm>
#include <concepts>
#include <type_traits>

//==============================================================================
// ALGORITHM
//==============================================================================
// Implementation of a subset of <algorithm> which is compatible with CUDA.
// See https://en.cppreference.com/w/cpp/algorithm for details.
// The following functions are implemented:
//  clamp
//  copy
//  fill_n
//  fill
//  is_sorted
//  max
//  max_element
//  min

namespace um2
{

//==============================================================================
// fill_n
//==============================================================================
// Reduces to a memset when possible.

template <class T, class Size>
HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::fill_n
fill_n(T * first, Size n, T const & value) noexcept -> T *
{
  for (; n > 0; ++first, --n) {
    *first = value;
  }
  return first;
}

//==============================================================================
// fill
//==============================================================================

// CUDA doesn't differentiate between random access and forward iterators, so
// we can't overload on that. We'll just use the forward iterator version.
#if !UM2_USE_CUDA

template <std::random_access_iterator RandomIt, class T>
HOST constexpr void
fill(RandomIt first, RandomIt last, T const & value) noexcept
{
  fill_n(first, last - first, value);
}

template <std::forward_iterator ForwardIt, class T>
HOST constexpr void
fill(ForwardIt first, ForwardIt last, T const & value) noexcept
{
  for (; first != last; ++first) {
    *first = value;
  }
}

#else

template <std::forward_iterator ForwardIt, class T>
HOSTDEV constexpr void
fill(ForwardIt first, ForwardIt last, T const & value) noexcept
{
  for (; first != last; ++first) {
    *first = value;
  }
}

#endif

//==============================================================================
// is_sorted
//==============================================================================

template <class ForwardIt>
PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::is_sorted
is_sorted(ForwardIt first, ForwardIt last) noexcept -> bool
{
  if (first == last) {
    return true;
  }

  auto next = first;
  ++next;

  while (next != last) {
    if (*next < *first) {
      return false;
    }
    ++first;
    ++next;
  }

  return true;
}

//==============================================================================
// max
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
max(T const & a, T const & b) noexcept -> T const &
{
  return a < b ? b : a;
}

//==============================================================================
// max_element
//==============================================================================

template <class ForwardIt>
PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::max_element
max_element(ForwardIt first, ForwardIt last) noexcept -> ForwardIt
{
  if (first != last) {
    ForwardIt i = first;
    while (++i != last) {
      if (*first < *i) {
        first = i;
      }
    }
  }
  return first;
}

//==============================================================================
// min
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
min(T const & a, T const & b) noexcept -> T const &
{
  return b < a ? b : a;
}

} // namespace um2
