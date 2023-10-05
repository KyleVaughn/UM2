#pragma once

#include <um2/common/Color.hpp>
#include <um2/common/ShortString.hpp>
#include <um2/physics/CrossSection.hpp>

namespace um2
{

struct Material {

  ShortString name;
  Color color;
  CrossSection xs;

  // ---------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------

  constexpr Material() noexcept = default;

  HOSTDEV constexpr Material(ShortString const & name_in, Color color_in) noexcept
      : name(name_in),
        color(color_in)
  {
  }

  HOSTDEV constexpr Material(ShortString const & name_in,
                             ShortString const & color_in) noexcept
      : name(name_in),
        color(color_in)
  {
  }

  template <uint64_t M, uint64_t N>
  HOSTDEV constexpr Material(char const (&name_in)[M], char const (&color_in)[N]) noexcept
      : name(name_in),
        color(color_in)
  {
  }
};

PURE HOSTDEV constexpr auto
operator==(Material const & lhs, Material const & rhs) -> bool
{
  return lhs.color == rhs.color && lhs.name == rhs.name;
}

PURE HOSTDEV constexpr auto
operator!=(Material const & lhs, Material const & rhs) -> bool
{
  return !(lhs == rhs);
}

} // namespace um2
