#pragma once

#include <um2/common/color.hpp>
#include <um2/physics/nuclide.hpp>
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
  String _name{};
  Color _color{};
  F _temperature{};         // [K]
  F _density{};             // [g/cm^3]
  Vector<F> _num_density;   // [atoms/b-cm]
  Vector<I> _zaid;          // ZZAAA

public :
  //======================================================================
  // Constructors
  //======================================================================

  constexpr Material() noexcept = default;

  //======================================================================
  // Setters and Getters
  //======================================================================

  void setName(String const & name) noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getName() const noexcept -> String const &
  {
    return _name;
  }

  constexpr void
  setColor(Color const & color) noexcept
  {
    _color = color;
  }

  constexpr void
  setTemperature(F temperature) noexcept
  {
    _temperature = temperature;
  }

  constexpr void
  setDensity(F density) noexcept
  {
    _density = density;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getColor() const noexcept -> Color
  {
    return _color;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getTemperature() const noexcept -> F
  {
    return _temperature;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getDensity() const noexcept -> F
  {
    return _density;
  }

///  PURE HOSTDEV [[nodiscard]] constexpr auto
///  numDensities() noexcept -> Vector<F> &
///  {
///    return _num_density;
///  }
///
///  PURE HOSTDEV [[nodiscard]] constexpr auto
///  numDensities() const noexcept -> Vector<F> const &
///  {
///    return _num_density;
///  }

  //======================================================================
  // Methods
  //======================================================================

  void
  validate() const noexcept;

  void
  addNuclide(I zaid, F num_density) noexcept;

  void
  addNuclide(String const & symbol, F num_density) noexcept;

};

} // namespace um2
