#pragma once

#include <um2/common/color.hpp>
#include <um2/stdlib/string.hpp>

//======================================================================
// MATERIAL
//======================================================================
//
// NOTE:
// The color is used for visualization and in Gmsh related operations.
// Since many CAD file formats do not support material assignment, but
// they do support color assignment, we use color as a proxy for
// material in I/O operations.

namespace um2
{

class Material
{

  String _name;
  Color _color;
  Vector<I> _zaid; // ZZAAA (Z = atomic number, A = atomic mass number)
  Vector<F> _

      public :
      //======================================================================
      // Constructors
      //======================================================================

      constexpr Material() noexcept = default;

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
  xs() noexcept -> CrossSection &
  {
    return _xs;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  xs() const noexcept -> CrossSection const &
  {
    return _xs;
  }
};

} // namespace um2
