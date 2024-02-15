#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/assert.hpp>

#include <cstring> // memmove
#include <cstdio> // printf
#include <iterator> // std::iterator_traits
#include <type_traits>

namespace um2
{

// Define a few type traits to check if we can use memmove or if we need to use a loop.

// https://github.com/llvm/llvm-project/blob/main/libcxx/include/__type_traits/is_always_bitcastable.h
template <class From, class To>
struct IsAlwaysBitcastable
{
  using UnqualFrom = std::remove_cv_t<From>;
  using UnqualTo = std::remove_cv_t<To>;
  static constexpr bool value = 
    (std::is_same_v<UnqualFrom, UnqualTo> && std::is_trivially_copyable_v<UnqualFrom>) ||
    (sizeof(From) == sizeof(To) && 
     std::is_integral_v<From> && 
     std::is_integral_v<To> && 
     !std::is_same<UnqualTo, bool>::value
    );
};

template <class From, class To>
struct CanLowerCopyToMemmove
{
  static constexpr bool value = IsAlwaysBitcastable<From, To>::value &&
                                std::is_trivially_assignable_v<To &, From const &> &&
                                !std::is_volatile_v<From> && !std::is_volatile_v<To>;
};

template <class It>
HOSTDEV constexpr auto
copyLoop(It first, It last, It d_first) noexcept -> It
{
  while (first != last) {
    *d_first = *first;
    ++first;
    ++d_first;
  }
  return d_first;
}

// This should technically be allowed to reduce to memmove, but we impose an additional 
// restriction that the destination range must not overlap with the source range when
// the memmove optimization is used. This allows us to reduce further to a memcpy.
//
// The memcpy should be slightly faster than memmove, but the main reason we use the memcpy
// is simply because memmove is not allowed in CUDA device code. 
//
// If this implementation causes issues, we can always write our own memmove implementation
// (it's very simple) and use that instead.
template <class It>
HOSTDEV constexpr auto
copy(It first, It last, It d_first) noexcept -> It
{
  using T = typename std::iterator_traits<It>::value_type;
  if constexpr (CanLowerCopyToMemmove<T, T>::value) {
    // Ensure that the destination range is not within the source range.
    // first -------- last
    //       d_first -------- d_last
    ASSERT(!(d_first >= first && d_first < last));
    //        first -------- last
    // d_first -------- d_last
    ASSERT(!(first >= d_first && first < d_first + (last - first)));
    return static_cast<It>(memcpy(d_first, first, 
        static_cast<size_t>(last - first) * sizeof(T)));
  } else {
    return copyLoop(first, last, d_first);
  }
}

} // namespace um2
