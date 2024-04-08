#pragma once

#include <um2/common/color.hpp>
#include <um2/physics/nuclide.hpp>
#include <um2/physics/cross_section_library.hpp>
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
  Float _temperature{};       // [K]
  Float _density{};           // [g/cm^3]
  Vector<Float> _num_density; // [atoms/b-cm]
  Vector<Int> _zaid;        // ZZAAA
  XSec _xsec;

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr Material() noexcept = default;

  //======================================================================
  // Setters and Getters
  //======================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numNuclides() const noexcept -> Int
  {
    return _zaid.size();
  }

  constexpr void
  setName(String const & name) noexcept
  {    
    ASSERT(!name.empty());
    _name = name;    
  }

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
  setTemperature(Float temperature) noexcept
  {
    _temperature = temperature;
  }

  constexpr void
  setDensity(Float density) noexcept
  {
    _density = density;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getColor() const noexcept -> Color
  {
    return _color;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getTemperature() const noexcept -> Float
  {
    return _temperature;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getDensity() const noexcept -> Float
  {
    return _density;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numDensities() noexcept -> Vector<Float> &
  {
    return _num_density;
  }
  
  PURE HOSTDEV [[nodiscard]] constexpr auto
  numDensities() const noexcept -> Vector<Float> const &
  {
    return _num_density;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numDensity(Int i) noexcept -> Float &
  {
    return _num_density[i];
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  numDensity(Int i) const noexcept -> Float 
  {
    return _num_density[i];
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zaids() noexcept -> Vector<Int> &
  {
    return _zaid;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zaids() const noexcept -> Vector<Int> const &
  {
    return _zaid;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zaid(Int i) noexcept -> Int &
  {
    return _zaid[i];
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  zaid(Int i) const noexcept -> Int
  {
    return _zaid[i];
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xsec() noexcept -> XSec &
  {
    return _xsec;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  xsec() const noexcept -> XSec const &
  {
    return _xsec;
  }

  //======================================================================
  // Methods
  //======================================================================

  void
  validate() const noexcept;

  void
  addNuclide(Int zaid, Float num_density) noexcept;

  void
  addNuclide(String const & symbol, Float num_density) noexcept;
};

} // namespace um2
