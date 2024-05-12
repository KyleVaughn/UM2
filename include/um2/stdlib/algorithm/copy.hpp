#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/utility/is_pointer_in_range.hpp>

#include <cstring>  // memcpy
#include <iterator> // std::iterator_traits
#include <type_traits>

namespace um2
{

// Define a few type traits to check if we can use memmove or if we need to use a loop.
// https://github.com/llvm/llvm-project/blob/main/libcxx/include/__type_traits/is_always_bitcastable.h
template <class From, class To>
struct IsAlwaysBitcastable {
  using UnqualFrom = std::remove_cv_t<From>;
  using UnqualTo = std::remove_cv_t<To>;
  static constexpr bool value =
      (std::is_same_v<UnqualFrom, UnqualTo> &&
       std::is_trivially_copyable_v<UnqualFrom>) ||
      (sizeof(From) == sizeof(To) && std::is_integral_v<From> && std::is_integral_v<To> &&
       !std::is_same_v<UnqualTo, bool>);
};

template <class From, class To>
struct CanLowerCopyToMemmove {
  static constexpr bool value = IsAlwaysBitcastable<From, To>::value &&
                                std::is_trivially_assignable_v<To &, From const &> &&
                                !std::is_volatile_v<From> && !std::is_volatile_v<To>;
};

template <class InputIt, class OutputIt>
HOSTDEV constexpr auto
copyLoop(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
{
  while (first != last) {
    *d_first = *first;
    ++first;
    ++d_first;
  }
  return d_first;
}

template <class InputIt, class OutputIt>
HOSTDEV constexpr auto
copy(InputIt first, InputIt last, OutputIt d_first) noexcept -> OutputIt
{
  using InT = typename std::iterator_traits<InputIt>::value_type;
  using OutT = typename std::iterator_traits<OutputIt>::value_type;
  if constexpr (CanLowerCopyToMemmove<InT, OutT>::value) {
    // Since d_first in [first, last) is undefined, we can do better than memmove,
    // we can use memcpy.
    auto const n = static_cast<size_t>(last - first);
    ASSERT(!is_pointer_in_range(first, last, d_first));
    ASSERT(!is_pointer_in_range(first, last, d_first + n));
    if (std::is_constant_evaluated()) {
      return copyLoop(first, last, d_first);
    }
    return static_cast<OutputIt>(memcpy(d_first, first, n * sizeof(InT))) + n;
  } else {
    return copyLoop(first, last, d_first);
  }
}

} // namespace um2
