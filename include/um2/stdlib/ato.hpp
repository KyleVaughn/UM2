#pragma once

#include <um2/config.hpp>

#include <cstdlib> // atoi, atof

//==============================================================================
// Template atoi, atof, etc. to ato<T> for ease of use
//==============================================================================

namespace um2
{

template <typename T>
PURE inline auto
ato(char const * /*s*/) noexcept -> T
{
  static_assert(always_false<T>, "Unsupported type");
}

template <>
PURE inline auto
ato<int16_t>(char const * s) noexcept -> int16_t
{
  return static_cast<int16_t>(std::atoi(s));
}

template <>
PURE inline auto
ato<int32_t>(char const * s) noexcept -> int32_t
{
  return std::atoi(s);
}

template <>
PURE inline auto
ato<int64_t>(char const * s) noexcept -> int64_t
{
  return std::atol(s);
}

template <>
PURE inline auto
ato<uint16_t>(char const * s) noexcept -> uint16_t
{
  return static_cast<uint16_t>(std::atoi(s));
}

template <>
PURE inline auto
ato<uint32_t>(char const * s) noexcept -> uint32_t
{
  return static_cast<uint32_t>(std::atol(s));
}

template <>
PURE inline auto
ato<uint64_t>(char const * s) noexcept -> uint64_t
{
  return static_cast<uint64_t>(std::atoll(s));
}

template <>
PURE inline auto
ato<float>(char const * s) noexcept -> float
{
  return static_cast<float>(std::atof(s));
}

template <>
PURE inline auto
ato<double>(char const * s) noexcept -> double
{
  return std::atof(s);
}

} // namespace um2
