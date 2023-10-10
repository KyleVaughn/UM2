#pragma once

#include <um2/config.hpp>

#include <um2/common/ShortString.hpp>

#include <concepts>

namespace um2
{

//==============================================================================
// COLOR
//==============================================================================
// A 4 byte RGBA color.
// Little endian: 0xAABBGGRR
// Common colors are defined in the .inl file

struct Color {

  struct RGBA {
    uint8_t r, g, b, a;
  };

  struct Rep {
    union {
      RGBA rgba;
      uint32_t u32; // 0xAABBGGRR
    };
  };

  Rep rep;

  //==============================================================================
  // Constructors
  //==============================================================================

  // Default to black (0, 0, 0, 255)
  HOSTDEV constexpr Color() noexcept;

  template <std::integral I>
  HOSTDEV constexpr Color(I r_in, I g_in, I b_in, I a_in = 255) noexcept;

  template <std::floating_point T>
  HOSTDEV constexpr Color(T r_in, T g_in, T b_in, T a_in = 1) noexcept;

  // We want to allow for implicit conversion for some cases
  // NOLINTBEGIN(google-explicit-constructor) justified
  HOSTDEV constexpr Color(ShortString const & name) noexcept;
  // NOLINTEND(google-explicit-constructor)

  //==============================================================================
  // Accessors
  //==============================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  r() const noexcept -> uint8_t;

  HOSTDEV [[nodiscard]] constexpr auto
  g() const noexcept -> uint8_t;

  HOSTDEV [[nodiscard]] constexpr auto
  b() const noexcept -> uint8_t;

  HOSTDEV [[nodiscard]] constexpr auto
  a() const noexcept -> uint8_t;
};

//==============================================================================
// Operators
//==============================================================================

CONST HOSTDEV constexpr auto
operator==(Color lhs, Color rhs) noexcept -> bool;

CONST HOSTDEV constexpr auto
operator!=(Color lhs, Color rhs) noexcept -> bool;

//==============================================================================
// Methods
//==============================================================================

PURE HOSTDEV constexpr auto
toColor(ShortString const & name) noexcept -> Color;

} // namespace um2

#include "Color.inl"
