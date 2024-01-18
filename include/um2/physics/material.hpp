#pragma once

#include <um2/common/color.hpp>
#include <um2/physics/cross_section.hpp>
#include <um2/stdlib/string.hpp>

//======================================================================
// MATERIAL
//======================================================================
// A physical material with a name, color, and multi-group cross section.
// The color is used for visualization and in Gmsh related operations.
// Since many CAD file formats do not support material assignment, but
// they do support color assignment, we use color as a proxy for
// material in I/O operations.

namespace um2
{

template <std::floating_point T>
class Material
{

  String _name;
  Color _color;
  CrossSection<T> _xs;

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr Material() noexcept = default;

  HOSTDEV constexpr Material(String name, Color color) noexcept
      : _name(um2::move(name)),
        _color(color)
  {
  }

  //======================================================================
  // Accessors
  //======================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  name() noexcept -> String &
  {
    return _name;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  name() const noexcept -> String const &
  {
    return _name;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  color() noexcept -> Color &
  {
    return _color;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  color() const noexcept -> Color const &
  {
    return _color;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  xs() noexcept -> CrossSection<T> &
  {
    return _xs;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  xs() const noexcept -> CrossSection<T> const &
  {
    return _xs;
  }
};

} // namespace um2
