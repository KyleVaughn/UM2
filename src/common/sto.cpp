#include <um2/common/sto.hpp>

namespace um2
{

template <>
UM2_PURE auto sto<int16_t>(std::string const & s) -> int16_t
{
  return static_cast<int16_t>(std::stoi(s));
}

template <>
UM2_PURE auto sto<int32_t>(std::string const & s) -> int32_t
{
  return std::stoi(s);
}

template <>
UM2_PURE auto sto<int64_t>(std::string const & s) -> int64_t
{
  return std::stol(s);
}

template <>
UM2_PURE auto sto<uint16_t>(std::string const & s) -> uint16_t
{
  return static_cast<uint16_t>(std::stoul(s));
}

template <>
UM2_PURE auto sto<uint32_t>(std::string const & s) -> uint32_t
{
  return static_cast<uint32_t>(std::stoul(s));
}

template <>
UM2_PURE auto sto<uint64_t>(std::string const & s) -> uint64_t
{
  return std::stoul(s);
}

template <>
UM2_PURE auto sto<float>(std::string const & s) -> float
{
  return std::stof(s);
}

template <>
UM2_PURE auto sto<double>(std::string const & s) -> double
{
  return std::stod(s);
}

} // namespace um2
