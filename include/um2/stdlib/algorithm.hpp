#pragma once

#include <um2/stdlib/utility.hpp>

#include <algorithm>

//==============================================================================
// ALGORITHM
//==============================================================================
// Implementation of a subset of <algorithm> which is compatible with CUDA.
// See https://en.cppreference.com/w/cpp/algorithm for details.
// The following functions are implemented:
//  clamp
//  copy
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

template <typename T>
HOSTDEV constexpr auto
clamp(T const & v, T const & lo, T const & hi) noexcept -> T
{
  return v < lo ? lo : (hi < v ? hi : v);
}

//==============================================================================
// copy
//==============================================================================

#ifndef __CUDA_ARCH__

// gcc seems to have a bug that causes it to generate a call to memmove that
// is out of bounds when using std::copy with -O3. This is a workaround.
// template <typename InputIt, typename OutputIt>
// HOST constexpr auto
// copy(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
//{
//  while (first != last) {
//    *d_first = *first;
//    ++first;
//    ++d_first;
//  }
//  return d_first;
//}

template <typename InputIt, typename OutputIt>
HOST constexpr auto
copy(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
{
  return std::copy(first, last, d_first);
}

#else

template <typename InputIt, typename OutputIt>
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
// fill
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename ForwardIt, typename T>
HOST constexpr void
fill(ForwardIt first, ForwardIt last, T const & value) noexcept
{
  std::fill(first, last, value);
}

#else

template <typename ForwardIt, typename T>
DEVICE constexpr void
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

// NOLINTBEGIN(readability-identifier-naming) match std::is_sorted
#ifndef __CUDA_ARCH__

template <typename ForwardIt>
PURE HOST constexpr auto
is_sorted(ForwardIt first, ForwardIt last) noexcept -> bool
{
  return std::is_sorted(first, last);
}

#else

template <typename ForwardIt>
PURE DEVICE constexpr auto
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

#endif
// NOLINTEND(readability-identifier-naming)

//==============================================================================
// max
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
CONST HOST constexpr auto
max(T x, T y) noexcept -> T
{
  return std::max(x, y);
}

#else

CONST DEVICE constexpr auto
max(float x, float y) noexcept -> float
{
  return ::fmaxf(x, y);
}

CONST DEVICE constexpr auto
max(double x, double y) noexcept -> double
{
  return ::fmax(x, y);
}

template <std::integral T>
CONST DEVICE constexpr auto
max(T x, T y) noexcept -> T
{
  return ::max(x, y);
}

#endif

//==============================================================================
// max_element
//==============================================================================

// NOLINTBEGIN(readability-identifier-naming) match std::max_element
#ifndef __CUDA_ARCH__

template <typename ForwardIt>
CONST HOST constexpr auto
max_element(ForwardIt first, ForwardIt last) noexcept -> ForwardIt
{
  return std::max_element(first, last);
}

#else

template <typename ForwardIt>
CONST DEVICE constexpr auto
max_element(ForwardIt first, ForwardIt last) noexcept -> ForwardIt
{
  if (first == last) {
    return last;
  }

  auto largest = first;
  ++first;

  while (first != last) {
    if (*largest < *first) {
      largest = first;
    }
    ++first;
  }

  return largest;
}

#endif
// NOLINTEND(readability-identifier-naming)

//==============================================================================
// min
//==============================================================================

#ifndef __CUDA_ARCH__

template <typename T>
CONST HOST constexpr auto
min(T x, T y) noexcept -> T
{
  return std::min(x, y);
}

#else

CONST DEVICE constexpr auto
min(float x, float y) noexcept -> float
{
  return ::fminf(x, y);
}

CONST DEVICE constexpr auto
min(double x, double y) noexcept -> double
{
  return ::fmin(x, y);
}

template <std::integral T>
CONST DEVICE constexpr auto
min(T x, T y) noexcept -> T
{
  return ::min(x, y);
}

#endif

} // namespace um2
