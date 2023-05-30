#pragma once

#include <um2/common/config.hpp>
#include <um2/common/string.hpp>

#include <concepts>

namespace um2
{

struct Color {

  uint8_t r, g, b, a;

  // -- Constructors --

  // Default to black (0, 0, 0, 255)
  UM2_HOSTDEV constexpr Color() noexcept;

  // From RGB, set alpha to 255
  template <std::integral I>
  UM2_HOSTDEV constexpr Color(I r_in, I g_in, I b_in, I a_in = 255) noexcept;

  // From floating point RGB, set alpha to 1.0
  template <std::floating_point T>
  UM2_HOSTDEV constexpr Color(T r_in, T g_in, T b_in, T a_in = 1) noexcept;

  // From a named color (see function definition for list)
  explicit Color(String const & name) noexcept;

  template <size_t N>
  explicit Color(char const (&name)[N]) noexcept;
};

// Operators
// -----------------------------------------------------------------------------

// If the native endianness is little, then rgba is stored in memory as
// abgr. This means that sorting by rgba is equivalent to sorting by abgr.

UM2_CONST UM2_HOSTDEV constexpr auto operator==(Color lhs, Color rhs) noexcept -> bool;

UM2_CONST UM2_HOSTDEV constexpr auto operator!=(Color lhs, Color rhs) noexcept -> bool;

UM2_CONST UM2_HOSTDEV constexpr auto operator<(Color lhs, Color rhs) noexcept -> bool;

// Methods
// -----------------------------------------------------------------------------

UM2_PURE auto toColor(String const & name) noexcept -> Color;

} // namespace um2

#include "color.inl"
