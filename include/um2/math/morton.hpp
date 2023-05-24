#pragma once

#include <um2/common/config.hpp>

#include <concepts>

#if defined(__BMI2__) && !defined(__CUDA_ARCH__)
#  include <immintrin.h> // _pdep_u64, _pext_u64, _pdep_u32, _pext_u32
#endif

namespace um2
{

// In N dimensions with an X bits morton code, the max bits that may be used to
// represent a coordinate without loss of precision is X / N.

template <std::unsigned_integral U>
static constexpr U morton_max_2d_coord = (static_cast<U>(1) << (4 * sizeof(U))) - 1;

template <std::unsigned_integral U>
static constexpr U morton_max_3d_coord = (static_cast<U>(1) << (8 * sizeof(U) / 3)) - 1;

#if defined(__BMI2__) && !defined(__CUDA_ARCH__)

static inline auto pdep(uint32_t source, uint32_t mask) noexcept -> uint32_t
{
  return _pdep_u32(source, mask);
}

static inline auto pdep(uint64_t source, uint64_t mask) noexcept -> uint64_t
{
  return _pdep_u64(source, mask);
}

static inline auto pext(uint32_t source, uint32_t mask) noexcept -> uint32_t
{
  return _pext_u32(source, mask);
}

static inline auto pext(uint64_t source, uint64_t mask) noexcept -> uint64_t
{
  return _pext_u64(source, mask);
}

template <std::unsigned_integral U>
static constexpr U bmi_2d_x_mask = static_cast<U>(0x5555555555555555);

template <std::unsigned_integral U>
static constexpr U bmi_2d_y_mask = static_cast<U>(0xAAAAAAAAAAAAAAAA);

template <std::unsigned_integral U>
UM2_NDEBUG_CONST inline U morton_encode(U const x, U const y)
{
  assert(x <= morton_max_2d_coord<U> && y <= morton_max_2d_coord<U>);
  return pdep(x, bmi_2d_x_mask<U>) | pdep(y, bmi_2d_y_mask<U>);
}

template <std::unsigned_integral U>
inline void morton_decode(U const morton, U & x, U & y)
{
  x = pext(morton, bmi_2d_x_mask<U>);
  y = pext(morton, bmi_2d_y_mask<U>);
}

template <std::unsigned_integral U>
static constexpr U bmi_3d_x_mask = static_cast<U>(0x9249249249249249);

template <std::unsigned_integral U>
static constexpr U bmi_3d_y_mask = static_cast<U>(0x2492492492492492);

template <std::unsigned_integral U>
static constexpr U bmi_3d_z_mask = static_cast<U>(0x4924924924924924);

template <std::unsigned_integral U>
UM2_NDEBUG_CONST inline U morton_encode(U const x, U const y, U const z)
{
  assert(x <= morton_max_3d_coord<U> && y <= morton_max_3d_coord<U> &&
         z <= morton_max_3d_coord<U>);
  return pdep(x, bmi_3d_x_mask<U>) | pdep(y, bmi_3d_y_mask<U>) |
         pdep(z, bmi_3d_z_mask<U>);
}

template <std::unsigned_integral U>
inline void morton_decode(U const morton, U & x, U & y, U & z)
{
  x = pext(morton, bmi_3d_x_mask<U>);
  y = pext(morton, bmi_3d_y_mask<U>);
  z = pext(morton, bmi_3d_z_mask<U>);
}

#else // this branch if !defined(__BMI2__) || defined(__CUDA_ARCH__)
// This is the fallback implementation of morton encoding/decoding that
// mimics the behavior of the BMI2 intrinsics.

UM2_NDEBUG_CONST UM2_HOSTDEV static constexpr auto pdep_u32_0x55555555(uint32_t x)
    -> uint32_t
{
  assert(x <= morton_max_2d_coord<uint32_t>);
  x = (x | (x << 8)) & 0x00ff00ff;
  x = (x | (x << 4)) & 0x0f0f0f0f;
  x = (x | (x << 2)) & 0x33333333;
  x = (x | (x << 1)) & 0x55555555;
  return x;
}

UM2_NDEBUG_CONST UM2_HOSTDEV static constexpr auto pdep_u64_0x5555555555555555(uint64_t x)
    -> uint64_t
{
  assert(x <= morton_max_2d_coord<uint64_t>);
  x = (x | (x << 16)) & 0x0000ffff0000ffff;
  x = (x | (x << 8)) & 0x00ff00ff00ff00ff;
  x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0f;
  x = (x | (x << 2)) & 0x3333333333333333;
  x = (x | (x << 1)) & 0x5555555555555555;
  return x;
}

UM2_NDEBUG_CONST UM2_HOSTDEV constexpr auto morton_encode(uint32_t const x,
                                                          uint32_t const y) -> uint32_t
{
  return pdep_u32_0x55555555(x) | (pdep_u32_0x55555555(y) << 1);
}

UM2_NDEBUG_CONST UM2_HOSTDEV constexpr auto morton_encode(uint64_t const x,
                                                          uint64_t const y) -> uint64_t
{
  return pdep_u64_0x5555555555555555(x) | (pdep_u64_0x5555555555555555(y) << 1);
}

UM2_CONST UM2_HOSTDEV constexpr static auto pext_u32_0x55555555(uint32_t x) -> uint32_t
{
  x &= 0x55555555;
  x = (x ^ (x >> 1)) & 0x33333333;
  x = (x ^ (x >> 2)) & 0x0f0f0f0f;
  x = (x ^ (x >> 4)) & 0x00ff00ff;
  x = (x ^ (x >> 8)) & 0x0000ffff;
  return x;
}

UM2_CONST UM2_HOSTDEV constexpr static auto pext_u64_0x5555555555555555(uint64_t x)
    -> uint64_t
{
  x &= 0x5555555555555555;
  x = (x ^ (x >> 1)) & 0x3333333333333333;
  x = (x ^ (x >> 2)) & 0x0f0f0f0f0f0f0f0f;
  x = (x ^ (x >> 4)) & 0x00ff00ff00ff00ff;
  x = (x ^ (x >> 8)) & 0x0000ffff0000ffff;
  x = (x ^ (x >> 16)) & 0x00000000ffffffff;
  return x;
}

UM2_HOSTDEV constexpr void morton_decode(uint32_t const morton, uint32_t & x,
                                         uint32_t & y)
{
  x = pext_u32_0x55555555(morton);
  y = pext_u32_0x55555555(morton >> 1);
}

UM2_HOSTDEV constexpr void morton_decode(uint64_t const morton, uint64_t & x,
                                         uint64_t & y)
{
  x = pext_u64_0x5555555555555555(morton);
  y = pext_u64_0x5555555555555555(morton >> 1);
}

UM2_NDEBUG_CONST UM2_HOSTDEV static constexpr auto pdep_u32_0x92492492(uint32_t x)
    -> uint32_t
{
  assert(x <= morton_max_3d_coord<uint32_t>);
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  x = (x | (x << 2)) & 0x09249249;
  return x;
}

UM2_NDEBUG_CONST UM2_HOSTDEV static constexpr auto pdep_u64_0x9249249249249249(uint64_t x)
    -> uint64_t
{
  assert(x <= morton_max_3d_coord<uint64_t>);
  x = (x | (x << 32)) & 0x001f00000000ffff;
  x = (x | (x << 16)) & 0x001f0000ff0000ff;
  x = (x | (x << 8)) & 0x100f00f00f00f00f;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3;
  x = (x | (x << 2)) & 0x1249249249249249;
  return x;
}

UM2_NDEBUG_CONST UM2_HOSTDEV constexpr auto
morton_encode(uint32_t const x, uint32_t const y, uint32_t const z) -> uint32_t
{
  return pdep_u32_0x92492492(x) | (pdep_u32_0x92492492(y) << 1) |
         (pdep_u32_0x92492492(z) << 2);
}

UM2_NDEBUG_CONST UM2_HOSTDEV constexpr auto
morton_encode(uint64_t const x, uint64_t const y, uint64_t const z) -> uint64_t
{
  return pdep_u64_0x9249249249249249(x) | (pdep_u64_0x9249249249249249(y) << 1) |
         (pdep_u64_0x9249249249249249(z) << 2);
}

UM2_CONST UM2_HOSTDEV constexpr static auto pext_u32_0x92492492(uint32_t x) -> uint32_t
{
  x &= 0x09249249;
  x = (x ^ (x >> 2)) & 0x030c30c3;
  x = (x ^ (x >> 4)) & 0x0300f00f;
  x = (x ^ (x >> 8)) & 0x030000ff;
  x = (x ^ (x >> 16)) & 0x000003ff;
  return x;
}

UM2_CONST UM2_HOSTDEV constexpr static auto pext_u64_0x9249249249249249(uint64_t x)
    -> uint64_t
{
  x &= 0x1249249249249249;
  x = (x ^ (x >> 2)) & 0x10c30c30c30c30c3;
  x = (x ^ (x >> 4)) & 0x100f00f00f00f00f;
  x = (x ^ (x >> 8)) & 0x001f0000ff0000ff;
  x = (x ^ (x >> 16)) & 0x001f00000000ffff;
  x = (x ^ (x >> 32)) & 0x00000000001fffff;
  return x;
}

UM2_HOSTDEV constexpr void morton_decode(uint32_t const morton, uint32_t & x,
                                         uint32_t & y, uint32_t & z)
{
  x = pext_u32_0x92492492(morton);
  y = pext_u32_0x92492492(morton >> 1);
  z = pext_u32_0x92492492(morton >> 2);
}

UM2_HOSTDEV constexpr void morton_decode(uint64_t const morton, uint64_t & x,
                                         uint64_t & y, uint64_t & z)
{
  x = pext_u64_0x9249249249249249(morton);
  y = pext_u64_0x9249249249249249(morton >> 1);
  z = pext_u64_0x9249249249249249(morton >> 2);
}

#endif // defined(__BMI2__) && !defined(__CUDA_ARCH__)

template <std::unsigned_integral U, std::floating_point T>
UM2_HOSTDEV auto normalized_morton_encode(T const x, T const y, T const xscale_inv,
                                          T const yscale_inv) -> U
{
  assert(x >= 0 && xscale_inv > 0);
  assert(y >= 0 && yscale_inv > 0);
  if constexpr (std::same_as<float, T> && std::same_as<uint64_t, U>) {
    static_assert(!sizeof(T), "uint64_t -> float conversion can be lossy");
  }
  U const max_coord = morton_max_2d_coord<U>;
  U const x_m = static_cast<U>(x * xscale_inv * static_cast<T>(max_coord));
  U const y_m = static_cast<U>(y * yscale_inv * static_cast<T>(max_coord));
  return morton_encode(x_m, y_m);
}

template <std::unsigned_integral U, std::floating_point T>
UM2_HOSTDEV void normalized_morton_decode(U const morton, T & x, T & y, T const xscale,
                                          T const yscale)
{
  assert(xscale > 0 && yscale > 0);
  constexpr T mm_inv = static_cast<T>(1) / morton_max_2d_coord<U>;
  U x_m;
  U y_m;
  morton_decode(morton, x_m, y_m);
  x = static_cast<T>(x_m) * xscale * mm_inv;
  y = static_cast<T>(y_m) * yscale * mm_inv;
}

template <std::unsigned_integral U, std::floating_point T>
UM2_HOSTDEV auto normalized_morton_encode(T const x, T const y, T const z,
                                          T const xscale_inv, T const yscale_inv,
                                          T const zscale_inv) -> U
{
  assert(0 <= x && 0 < xscale_inv);
  assert(0 <= y && 0 < yscale_inv);
  assert(0 <= z && 0 < zscale_inv);
  U const x_m = static_cast<U>(x * xscale_inv * morton_max_3d_coord<U>);
  U const y_m = static_cast<U>(y * yscale_inv * morton_max_3d_coord<U>);
  U const z_m = static_cast<U>(z * zscale_inv * morton_max_3d_coord<U>);
  return morton_encode(x_m, y_m, z_m);
}

template <std::unsigned_integral U, std::floating_point T>
UM2_HOSTDEV void normalized_morton_decode(U const morton, T & x, T & y, T & z,
                                          T const xscale, T const yscale, T const zscale)
{
  assert(xscale > 0 && yscale > 0 && zscale > 0);
  constexpr T mm_inv = static_cast<T>(1) / morton_max_3d_coord<U>;
  U x_m;
  U y_m;
  U z_m;
  morton_decode(morton, x_m, y_m, z_m);
  x = static_cast<T>(x_m) * xscale * mm_inv;
  y = static_cast<T>(y_m) * yscale * mm_inv;
  z = static_cast<T>(z_m) * zscale * mm_inv;
}
} // namespace um2
