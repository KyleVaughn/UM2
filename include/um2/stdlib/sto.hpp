#pragma once

#include <um2/config.hpp>

#include <string> // std::string, std::stoi, std::stod, etc.

//==============================================================================
// Template stoi, stod, etc. to sto<T> for ease of use
//==============================================================================

namespace um2
{

template <typename T>
PURE inline auto
sto(std::string const & /*s*/) noexcept -> T
{
  static_assert(always_false<T>, "Unsupported type");
}

template <>
PURE inline auto
sto<int16_t>(std::string const & s) noexcept -> int16_t
{
  return static_cast<int16_t>(std::stoi(s));
}

template <>
PURE inline auto
sto<int32_t>(std::string const & s) noexcept -> int32_t
{
  return std::stoi(s);
}

template <>
PURE inline auto
sto<int64_t>(std::string const & s) noexcept -> int64_t
{
  return std::stol(s);
}

template <>
PURE inline auto
sto<uint16_t>(std::string const & s) noexcept -> uint16_t
{
  return static_cast<uint16_t>(std::stoul(s));
}

template <>
PURE inline auto
sto<uint32_t>(std::string const & s) noexcept -> uint32_t
{
  return static_cast<uint32_t>(std::stoul(s));
}

template <>
PURE inline auto
sto<uint64_t>(std::string const & s) noexcept -> uint64_t
{
  return std::stoul(s);
}

template <>
PURE inline auto
sto<float>(std::string const & s) noexcept -> float
{
  return std::stof(s);
}

template <>
PURE inline auto
sto<double>(std::string const & s) noexcept -> double
{
  return std::stod(s);
}

} // namespace um2
