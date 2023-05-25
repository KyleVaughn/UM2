#pragma once

#include <string> // std::string, std::stoi, std::stod, etc.
#include <um2/common/config.hpp>

namespace um2
{

template <typename T>
UM2_PURE auto sto(std::string const & /*s*/) noexcept -> T
{
  static_assert(!sizeof(T), "Unsupported type");
}

template <>
UM2_PURE auto sto<int16_t>(std::string const & s) noexcept -> int16_t;

template <>
UM2_PURE auto sto<int32_t>(std::string const & s) noexcept -> int32_t;

template <>
UM2_PURE auto sto<int64_t>(std::string const & s) noexcept -> int64_t;

template <>
UM2_PURE auto sto<uint16_t>(std::string const & s) noexcept -> uint16_t;

template <>
UM2_PURE auto sto<uint32_t>(std::string const & s) noexcept -> uint32_t;

template <>
UM2_PURE auto sto<uint64_t>(std::string const & s) noexcept -> uint64_t;

template <>
UM2_PURE auto sto<float>(std::string const & s) noexcept -> float;

template <>
UM2_PURE auto sto<double>(std::string const & s) noexcept -> double;

} // namespace um2
