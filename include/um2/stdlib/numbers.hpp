#pragma once

namespace um2
{

// NOLINTBEGIN(modernize-use-std-numbers)

template <class T>
inline constexpr T e = static_cast<T>(2.71828182845904523536);

template <class T>
inline constexpr T pi = static_cast<T>(3.14159265358979323846);

template <class T>
inline constexpr T pi_2 = static_cast<T>(1.57079632679489661923);

template <class T>
inline constexpr T pi_4 = static_cast<T>(0.785398163397448309616);

// NOLINTEND(modernize-use-std-numbers)

} // namespace um2
