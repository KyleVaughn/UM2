#pragma once

#include <um2/config.hpp>

#include <cstring> // memmove
#include <cstdio> // printf
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

template <class T>
HOSTDEV constexpr auto
copyLoop(T * first, T * last, T * d_first) noexcept -> T *
{
  while (first != last) {
    *d_first = *first;
    ++first;
    ++d_first;
  }
  return d_first;
}

// Reduce to memmove if possible.
template <class T>
requires (CanLowerCopyToMemmove<T, T>::value)
HOSTDEV constexpr auto
copy(T * first, T * last, T * d_first) noexcept -> T*
{
  return static_cast<T*>(std::memmove(d_first, first, 
        static_cast<size_t>(last - first) * sizeof(T)));
}

template <class T>
requires (!CanLowerCopyToMemmove<T, T>::value)
HOSTDEV constexpr auto
copy(T * first, T * last, T * d_first) noexcept -> T*
{
  return copyLoop(first, last, d_first);
}

} // namespace um2
