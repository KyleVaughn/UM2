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
// clamp
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
clamp(T const & v, T const & lo, T const & hi) noexcept -> T const &
{
  return v < lo ? lo : (hi < v ? hi : v);
}

//==============================================================================
// copy
//==============================================================================
// std::copy reduces to a memmove when possible. The mechanism for this is
// pretty complicated, so I would recommend just using std::copy and settling
// for a less performance copy on device. If you really need to optimize this,
// you could overload for pointers to fundamental types.

#ifndef __CUDA_ARCH__

template <class InputIt, class OutputIt>
HOST constexpr auto
copy(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
{
  return std::copy(first, last, d_first);
}

#else

// TODO(kcvaughn): Overload this for cases which reduce to a memmove.
// https://github.com/KyleVaughn/UM2/issues/141

template <class InputIt, class OutputIt>
DEVICE constexpr auto
copy(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
{
  while (first != last) {
    *d_first = *first;
    ++first;
    ++d_first;
  }
  return d_first;
}

#endif

//==============================================================================
// fill_n
//==============================================================================

template <class OutputIt, class Size, class T>
HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-identifier-naming) match std::fill_n
fill_n(OutputIt first, Size n, T const & value) noexcept -> OutputIt
{
  for (; n > 0; ++first, --n) {
    *first = value;
  }
  return first;
}

//==============================================================================
// fill
//==============================================================================

template <std::forward_iterator ForwardIt, class T>
HOSTDEV constexpr void
fill(ForwardIt first, ForwardIt last, T const & value) noexcept
{
  for (; first != last; ++first) {
    *first = value;
  }
}

template <std::random_access_iterator RandomIt, class T>
HOSTDEV constexpr void
fill(RandomIt first, RandomIt last, T const & value) noexcept
{
  fill_n(first, last - first, value);
}

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
