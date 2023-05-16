#pragma once

#include <bit>
#include <concepts>
#include <cstdint>

namespace um2
{

#ifndef __CUDA_ARCH__

constexpr uint8_t bit_ceil(int8_t x) { return std::bit_ceil(std::bit_cast<uint8_t>(x)); }

constexpr uint16_t bit_ceil(int16_t x) { return std::bit_ceil(std::bit_cast<uint16_t>(x)); }

constexpr uint32_t bit_ceil(int32_t x) { return std::bit_ceil(std::bit_cast<uint32_t>(x)); }

constexpr uint64_t bit_ceil(int64_t x) { return std::bit_ceil(std::bit_cast<uint64_t>(x)); }

template <std::unsigned_integral T>
constexpr T bit_ceil(T x)
{
  return std::bit_ceil(x);
}

#else // __CUDA_ARCH__

__device__ constexpr uint32_t bit_ceil(uint32_t x)
{
  if (x <= 1) {
    return x;
  } else {
    return 1 << (32 - __clz(x - 1));
  }
}

__device__ constexpr uint32_t bit_ceil(int32_t x) { return bit_ceil(static_cast<uint32_t>(x)); }

__device__ constexpr uint64_t bit_ceil(uint64_t x)
{
  if (x <= 1) {
    return x;
  } else {
    return 1 << (64 - __clzll(x - 1));
  }
}

__device__ constexpr uint64_t bit_ceil(int64_t x) { return bit_ceil(static_cast<uint64_t>(x)); }

#endif // __CUDA_ARCH__

} // namespace um2
