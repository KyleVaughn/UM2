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
// _temperature: the temperatures at which the microscopic cross section data
//               is defined.
// _xs: the microscopic cross section data at each temperature.
//
// NOTE: _xs is basically a 2D array. We should use a better data structure if
//  performance becomes an issue.

namespace um2
{

class Nuclide
{
  I _zaid{};
  F _mass{};
  Vector<F> _temperature;
  Vector<XSec> _xs;

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr Nuclide() noexcept = default;

  //======================================================================
  // Accessors
  //======================================================================

  [[nodiscard]] constexpr auto
  zaid() noexcept -> I &
  {
    return _zaid;
  }

  [[nodiscard]] constexpr auto
  zaid() const noexcept -> I const &
  {
    return _zaid;
  }

  [[nodiscard]] constexpr auto
  mass() noexcept -> F &
  {
    return _mass;
  }

  [[nodiscard]] constexpr auto
  mass() const noexcept -> F const &
  {
    return _mass;
  }

  [[nodiscard]] constexpr auto
  temperature() noexcept -> Vector<F> &
  {
    return _temperature;
  }

  [[nodiscard]] constexpr auto
  temperature() const noexcept -> Vector<F> const &
  {
    return _temperature;
  }

  [[nodiscard]] constexpr auto
  xs() noexcept -> Vector<XSec> &
  {
    return _xs;
  }

  [[nodiscard]] constexpr auto
  xs() const noexcept -> Vector<XSec> const &
  {
    return _xs;
  }

  //======================================================================
  // Methods
  //======================================================================

  void
  validate() const noexcept
  {
    ASSERT(_zaid > 0);
    for (auto const & x : _xs) {
      x.validate();
      ASSERT(!x.isMacroscopic())
    }
    ASSERT(_temperature.size() == _xs.size())
  }

}; // class Nuclide

//======================================================================
// Non-member functions
//======================================================================

// Convert from a string, like "U-235", to a ZAID.
auto
toZAID(String str) -> I;

} // namespace um2
