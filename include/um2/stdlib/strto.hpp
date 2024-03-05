#pragma once

#include <um2/config.hpp>

#include <cstdlib>

//==============================================================================
// STRTO
//==============================================================================
// Templated versions of strtof, strtod, etc. to strto<T> for ease of use.

namespace um2
{

template <typename T>
inline auto
strto(char const * /*str*/, char ** /*endptr*/) noexcept -> T
{
  static_assert(always_false<T>, "Unsupported type");
}

template <>
inline auto
strto<float>(char const * str, char ** endptr) noexcept -> float
{
  return std::strtof(str, endptr);
}

template <>
inline auto
strto<double>(char const * str, char ** endptr) noexcept -> double
{
  return std::strtod(str, endptr);
}

template <>
inline auto
strto<int32_t>(char const * str, char ** endptr) noexcept -> int32_t
{
  return static_cast<int32_t>(std::strtol(str, endptr, 10));
}

template <>
inline auto
strto<int64_t>(char const * str, char ** endptr) noexcept -> int64_t
{
  return std::strtol(str, endptr, 10);
}

} // namespace um2
