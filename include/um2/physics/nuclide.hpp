#pragma once

#include <um2/physics/cross_section.hpp>
#include <um2/stdlib/string.hpp>

//======================================================================
// NUCLIDE
//======================================================================
// Data structure for storing nuclide data.
//
// _zaid: the ZZAAA identifier for the nuclide.
//
// _mass: the atomic mass of the nuclide.
// _temperatures: the temperatures at which the microscopic cross section data
//               is defined.
// _xs: the microscopic cross section data at each temperature.
//
// NOTE: _xs is basically a 2D array. We should use a better data structure if
//  performance becomes an issue.

namespace um2
{

class Nuclide
{
  bool _is_fissile = false;
  I _zaid{};
  F _mass{};
  Vector<F> _temperatures;
  Vector<XSec> _xs;

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr Nuclide() noexcept = default;

  //======================================================================
  // Accessors
  //======================================================================

  PURE [[nodiscard]] constexpr auto
  isFissile() noexcept -> bool &
  {
    return _is_fissile;
  }

  PURE [[nodiscard]] constexpr auto
  isFissile() const noexcept -> bool const &
  {
    return _is_fissile;
  }

  PURE [[nodiscard]] constexpr auto
  zaid() noexcept -> I &
  {
    return _zaid;
  }

  PURE [[nodiscard]] constexpr auto
  zaid() const noexcept -> I const &
  {
    return _zaid;
  }

  PURE [[nodiscard]] constexpr auto
  mass() noexcept -> F &
  {
    return _mass;
  }

  PURE [[nodiscard]] constexpr auto
  mass() const noexcept -> F const &
  {
    return _mass;
  }

  PURE [[nodiscard]] constexpr auto
  temperatures() noexcept -> Vector<F> &
  {
    return _temperatures;
  }

  PURE [[nodiscard]] constexpr auto
  temperatures() const noexcept -> Vector<F> const &
  {
    return _temperatures;
  }

  PURE [[nodiscard]] constexpr auto
  xs() noexcept -> Vector<XSec> &
  {
    return _xs;
  }

  PURE [[nodiscard]] constexpr auto
  xs() const noexcept -> Vector<XSec> const &
  {
    return _xs;
  }

  //======================================================================
  // Methods
  //======================================================================

  void
  clear() noexcept;

  void
  validate() const noexcept;

}; // class Nuclide

//======================================================================
// Non-member functions
//======================================================================

// Convert from a string, like "U-235", to a ZAID.
auto
toZAID(String str) -> I;

} // namespace um2
