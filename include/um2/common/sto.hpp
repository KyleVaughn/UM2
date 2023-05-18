#pragma once

#include <um2/common/config.hpp>
#include <string> // std::string, std::stoi, std::stod, etc.

namespace um2
{

template <typename T>
UM2_PURE
auto sto(std::string const & /*s*/) -> T
{
  static_assert(!sizeof(T), "Unsupported type");
}

template <>
UM2_PURE
auto sto<int16_t>(std::string const & s) -> int16_t;

template <>
UM2_PURE
auto sto<int32_t>(std::string const & s) -> int32_t;

template <>
UM2_PURE
auto sto<int64_t>(std::string const & s) -> int64_t;

template <>
UM2_PURE
auto sto<uint16_t>(std::string const & s) -> uint16_t;

template <>
UM2_PURE
auto sto<uint32_t>(std::string const & s) -> uint32_t;

template <>
UM2_PURE
auto sto<uint64_t>(std::string const & s) -> uint64_t;

template <>
UM2_PURE
auto sto<float>(std::string const & s) -> float;

template <>
UM2_PURE
auto sto<double>(std::string const & s) -> double;

} // namespace um2
