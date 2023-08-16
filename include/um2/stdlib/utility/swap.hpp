#pragma once

#include <um2/config.hpp>

#include <utility>

namespace um2
{

// -----------------------------------------------------------------------------
// swap
// -----------------------------------------------------------------------------
// https://en.cppreference.com/w/cpp/utility/swap

using std::swap;

#ifdef __CUDA_ARCH__

template <class T>
  requires(std::is_trivially_move_constructible_v<T> &&
           std::is_trivially_move_assignable_v<T>)
DEVICE constexpr void swap(T & a, T & b) noexcept
{
  T tmp = um2::move(a);
  a = um2::move(b);
  b = um2::move(tmp);
}

#endif

} // namespace um2
