#pragma once

#include <um2/common/config.hpp>
#include <um2/common/string.hpp>

#include <concepts>

namespace um2
{

struct Color {

  uint8_t r{}, g{}, b{}, a{};

  // -- Constructors --

  // Default to black (0, 0, 0, 255)
  UM2_HOSTDEV constexpr Color();

  // From RGB, set alpha to 255
  template <std::integral I>
  UM2_HOSTDEV constexpr Color(I r_in, I g_in, I b_in, I a_in = 255);

  // From floating point RGB, set alpha to 1.0
  template <std::floating_point T>
  UM2_HOSTDEV constexpr Color(T r_in, T g_in, T b_in, T a_in = 1);

  // From a named color (see function definition for list)
  explicit Color(String const & name);

  template <size_t N>
  // NOLINTNEXTLINE(*-avoid-c-arrays)
  explicit Color(char const (&name)[N]);
};

// Operators
// -----------------------------------------------------------------------------

UM2_CONST UM2_HOSTDEV constexpr auto operator==(Color lhs, Color rhs) -> bool;

UM2_CONST UM2_HOSTDEV constexpr auto operator!=(Color lhs, Color rhs) -> bool;

UM2_CONST UM2_HOSTDEV constexpr auto operator<(Color lhs, Color rhs) -> bool;

// Methods
// -----------------------------------------------------------------------------

UM2_PURE auto to_rgba(String const & name) -> Color;

} // namespace um2

#include "color.inl"
