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
};

} // namespace um2
