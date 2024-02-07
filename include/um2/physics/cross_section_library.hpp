#pragma once

#include <um2/physics/nuclide.hpp>

//======================================================================
// CROSS SECTION LIBRARY
//======================================================================
// A multi-group cross section library.
//
// The cross section library is collection of nuclides, each of which
// have a microscopic XSec object. The nuclides are grouped by
// temperature.

namespace um2
{

// Some common cross section libraries.
namespace mpact
{
const String XSLIB_8G = "mpact8g_70s_v4.0m0_02232015.fmt";
const String XSLIB_51G = "mpact51g_71_v4.2m5_12062016_sph.fmt";
} // namespace mpact

class XSLibrary
{

  Vector<F> _group_bounds; // Energy bounds. size = numGroups()
  Vector<F> _chi;          // Fission spectrum. size = numGroups()
  Vector<Nuclide> _nuclides;

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr XSLibrary() noexcept = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  XSLibrary(String const & filename);

  //======================================================================
  // Accessors
  //======================================================================

  PURE [[nodiscard]] constexpr auto
  groupBounds() noexcept -> Vector<F> &
  {
    return _group_bounds;
  }

  PURE [[nodiscard]] constexpr auto
  groupBounds() const noexcept -> Vector<F> const &
  {
    return _group_bounds;
  }

  PURE [[nodiscard]] constexpr auto
  chi() noexcept -> Vector<F> &
  {
    return _chi;
  }

  PURE [[nodiscard]] constexpr auto
  chi() const noexcept -> Vector<F> const &
  {
    return _chi;
  }

  PURE [[nodiscard]] constexpr auto
  nuclides() noexcept -> Vector<Nuclide> &
  {
    return _nuclides;
  }

  PURE [[nodiscard]] constexpr auto
  nuclides() const noexcept -> Vector<Nuclide> const &
  {
    return _nuclides;
  }

  PURE [[nodiscard]] constexpr auto
  numGroups() const noexcept -> I
  {
    return _group_bounds.size();
  }

  //======================================================================
  // Methods
  //======================================================================
};

} // namespace um2
