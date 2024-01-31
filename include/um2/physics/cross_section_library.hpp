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

  [[nodiscard]] constexpr auto
  groupBounds() noexcept -> Vector<F> &
  {
    return _group_bounds;
  }

  [[nodiscard]] constexpr auto
  groupBounds() const noexcept -> Vector<F> const &
  {
    return _group_bounds;
  }

  [[nodiscard]] constexpr auto
  chi() noexcept -> Vector<F> &
  {
    return _chi;
  }

  [[nodiscard]] constexpr auto
  chi() const noexcept -> Vector<F> const &
  {
    return _chi;
  }

  [[nodiscard]] constexpr auto
  nuclides() noexcept -> Vector<Nuclide> &
  {
    return _nuclides;
  }

  [[nodiscard]] constexpr auto
  nuclides() const noexcept -> Vector<Nuclide> const &
  {
    return _nuclides;
  }

  [[nodiscard]] constexpr auto
  numGroups() const noexcept -> Size
  {
    return _group_bounds.size();
  }

  //======================================================================
  // Methods
  //======================================================================
};

} // namespace um2
