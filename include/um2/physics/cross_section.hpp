#pragma once

#include <um2/common/log.hpp>
#include <um2/math/stats.hpp>
#include <um2/stdlib/vector.hpp>

//======================================================================
// CROSS SECTION
//======================================================================
// A multi-group cross section that can represent macroscopic or
// microscopic cross sections.
//
// Units: [cm^-1] or [barns] depending on whether the cross section is
// macroscopic or microscopic.
//
// Currently, we only need the total cross section, but will add
// scattering, etc. later.

namespace um2
{

enum class XSecReduction {
  Max,
  Mean,
};

class XSec
{

  bool _is_macroscopic = false;
  Vector<F> _t; // Total cross section

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr XSec() noexcept = default;

  //======================================================================
  // Accessors
  //======================================================================

  PURE [[nodiscard]] constexpr auto
  isMacro() noexcept -> bool &
  {
    return _is_macroscopic;
  }

  PURE [[nodiscard]] constexpr auto
  isMacro() const noexcept -> bool
  {
    return _is_macroscopic;
  }

  PURE [[nodiscard]] constexpr auto
  t() noexcept -> Vector<F> &
  {
    return _t;
  }

  PURE [[nodiscard]] constexpr auto
  t() const noexcept -> Vector<F> const &
  {
    return _t;
  }

  PURE [[nodiscard]] constexpr auto
  t(I g) noexcept -> F &
  {
    return _t[g];
  }

  PURE [[nodiscard]] constexpr auto
  t(I g) const noexcept -> F
  {
    return _t[g];
  }

  PURE [[nodiscard]] constexpr auto
  numGroups() const noexcept -> I
  {
    return _t.size();
  }

  //======================================================================
  // Methods
  //======================================================================

  void
  validate() const noexcept;

  // Get the 1-group cross section
  PURE [[nodiscard]] auto
  collapse(XSecReduction strategy = XSecReduction::Mean) const noexcept -> XSec;
}; // class XS

} // namespace um2
