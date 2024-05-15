#pragma once

#include <um2/config.hpp>

#include <concepts> // std::same_as

// if To and From are the same type, return x
// otherwise, return static_cast<To>(x)
// Should result in no-op for same types
template <class To, class From>
CONST HOSTDEV constexpr auto
castIfNot(From const x) noexcept -> To
{
  if constexpr (std::same_as<To, From>) {
    return x;
  } else {
    return static_cast<To>(x);
  }
}
