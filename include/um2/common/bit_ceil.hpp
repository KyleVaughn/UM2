#pragma once

#include <bit>
#include <concepts>

namespace um2
{

#ifndef __CUDA_ARCH__
constexpr unsigned char bit_ceil(char x) { return std::bit_ceil(std::bit_cast<unsigned char>(x)); }

constexpr unsigned short bit_ceil(short x)
{
  return std::bit_ceil(std::bit_cast<unsigned short>(x));
}

constexpr unsigned int bit_ceil(int x) { return std::bit_ceil(std::bit_cast<unsigned int>(x)); }

constexpr unsigned long bit_ceil(long x) { return std::bit_ceil(std::bit_cast<unsigned long>(x)); }

constexpr unsigned long long bit_ceil(long long x)
{
  return std::bit_ceil(std::bit_cast<unsigned long long>(x));
}

template <std::unsigned_integral T>
constexpr T bit_ceil(T x)
{
  return std::bit_ceil(x);
}

#else // __CUDA_ARCH__

__device__ constexpr unsigned int bit_ceil(unsigned int x)
{
  if (x <= 1) {
    return x;
  } else {
    return 1 << (32 - __clz(x - 1));
  }
}

__device__ constexpr unsigned int bit_ceil(int x) { return bit_ceil(static_cast<unsigned int>(x)); }

#endif // __CUDA_ARCH__

} // namespace um2
