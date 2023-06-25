#pragma once

#include <um2/common/config.hpp>
#include <um2/common/string.hpp>

#include <concepts>

namespace um2
{

// -----------------------------------------------------------------------------
// COLOR
// -----------------------------------------------------------------------------
// A 4 byte RGBA color.

struct Color {

  union {
    struct {
      uint8_t r, g, b, a;
    };
    uint32_t rgba;
  };

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  // Default to black (0, 0, 0, 255)
  UM2_HOSTDEV constexpr Color() noexcept;

  template <std::integral I>
  UM2_HOSTDEV constexpr Color(I r_in, I g_in, I b_in, I a_in = 255) noexcept;

  template <std::floating_point T>
  UM2_HOSTDEV constexpr Color(T r_in, T g_in, T b_in, T a_in = 1) noexcept;

  template <size_t N>
  UM2_HOSTDEV constexpr explicit Color(char const (&name)[N]) noexcept;

  UM2_HOSTDEV constexpr explicit Color(String const & name) noexcept;
};

// -----------------------------------------------------------------------------
// Operators
// -----------------------------------------------------------------------------

UM2_CONST UM2_HOSTDEV constexpr auto
operator==(Color lhs, Color rhs) noexcept -> bool;

UM2_CONST UM2_HOSTDEV constexpr auto
operator!=(Color lhs, Color rhs) noexcept -> bool;

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

UM2_PURE UM2_HOSTDEV constexpr auto
toColor(String const & name) noexcept -> Color;

} // namespace um2

#include "color.inl"
