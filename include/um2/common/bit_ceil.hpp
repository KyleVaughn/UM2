#pragma once

#include <bit>
#include <concepts>

namespace um2 {

constexpr unsigned char bit_ceil(char x) { return std::bit_ceil(std::bit_cast<unsigned char>(x)); }
constexpr unsigned short bit_ceil(short x) { return std::bit_ceil(std::bit_cast<unsigned short>(x)); }
constexpr unsigned int bit_ceil(int x) { return std::bit_ceil(std::bit_cast<unsigned int>(x)); }
constexpr unsigned long bit_ceil(long x) { return std::bit_ceil(std::bit_cast<unsigned long>(x)); }
constexpr unsigned long long bit_ceil(long long x) { return std::bit_ceil(std::bit_cast<unsigned long long>(x)); }
template <std::unsigned_integral T> constexpr T bit_ceil(T x) { return std::bit_ceil(x); }

} // namespace um2
