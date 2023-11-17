#pragma once

#include <um2/common/color.hpp>
#include <um2/common/shortstring.hpp>
#include <um2/physics/cross_section.hpp>

namespace um2
{

template <std::floating_point T>
struct Material {

  ShortString name;
  Color color;
  CrossSection<T> xs;

  //======================================================================
  // Constructors
  //======================================================================

  constexpr Material() noexcept = default;

  HOSTDEV constexpr Material(ShortString const & name_in, Color color_in) noexcept
      : name(name_in),
        color(color_in)
  {
  }

  // TODO(kcvaughn): implicitly convert for name and color

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

  //======================================================================
  // Methods
  //======================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  hasTotalCrossSection() const noexcept -> bool
  {
    return !xs.t.empty();
  }
};

// PURE HOSTDEV constexpr auto
// operator==(Material const & lhs, Material const & rhs) -> bool
// {
//   return lhs.color == rhs.color && lhs.name == rhs.name;
// }
//
// PURE HOSTDEV constexpr auto
// operator!=(Material const & lhs, Material const & rhs) -> bool
// {
//   return !(lhs == rhs);
// }

} // namespace um2
