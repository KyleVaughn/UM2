#pragma once

#include <um2/config.hpp>
#include <um2/stdlib/utility/move.hpp>

#include <cstdlib> // std::size_t
#include <type_traits> // std::is_trivially_move_constructible_v, std::is_trivially_move_assignable_v

namespace um2
{

template <class T>
  requires(std::is_trivially_move_constructible_v<T> &&
           std::is_trivially_move_assignable_v<T>)
HOSTDEV constexpr void swap(T & a, T & b) noexcept
{
  T tmp = um2::move(a);
  a = um2::move(b);
  b = um2::move(tmp);
}

template <class T, size_t N>
HOSTDEV constexpr void
swap(T (&a)[N], T (&b)[N]) noexcept
{
  for (size_t i = 0; i < N; ++i) {
    um2::swap(a[i], b[i]);
  }
}

} // namespace um2
