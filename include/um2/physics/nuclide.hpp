#pragma once

#include <um2/physics/cross_section.hpp>
#include <um2/stdlib/string.hpp>

//======================================================================
// NUCLIDE
//======================================================================
// Data structure for storing nuclide data.
//
// _zaid: the ZZAAA identifier for the nuclide.
// _xs: the microscopic cross section data for the nuclide.

namespace um2
{

class Nuclide
{
  I _zaid{};
  CrossSection _xs;

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr Nuclide() noexcept = default;

  //======================================================================
  // Accessors
  //======================================================================

  [[nodiscard]] constexpr auto zaid() noexcept -> I &
  {
    return _zaid;
  }

  [[nodiscard]] constexpr auto zaid() const noexcept -> I const &
  {
    return _zaid;
  };

  [[nodiscard]] constexpr auto xs() noexcept -> CrossSection &
  {
    return _xs;
  }

  [[nodiscard]] constexpr auto xs() const noexcept -> CrossSection const &
  {
    return _xs;
  }

  //======================================================================
  // Methods
  //======================================================================

  void validate() const noexcept
  {
    ASSERT(_zaid > 0);
    _xs.validate();
    ASSERT(!_xs.isMacroscopic());
  }

}; // class Nuclide

//======================================================================
// Non-member functions
//======================================================================

// Convert from a string, like "U-235", to a ZAID.
auto 
toZAID(String str) -> I;

} // namespace um2
