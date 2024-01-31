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

  [[nodiscard]] constexpr auto
  isMacroscopic() const noexcept -> bool
  {
    return _is_macroscopic;
  }

  [[nodiscard]] constexpr auto
  numGroups() const noexcept -> Size
  {
    return _t.size();
  }

  [[nodiscard]] constexpr auto
  t() noexcept -> Vector<F> &
  {
    return _t;
  }

  [[nodiscard]] constexpr auto
  t() const noexcept -> Vector<F> const &
  {
    return _t;
  }

  //======================================================================
  // Methods
  //======================================================================

  void
  validate() const noexcept
  {
    if (_t.empty()) {
      LOG_ERROR("Cross section has an empty total XS vector");
    }
    for (auto const & t_i : _t) {
      if (t_i < 0) {
        LOG_ERROR("Cross section has a negative total XS in one or more groups");
      }
    }
  }

  [[nodiscard]] auto constexpr get1GroupTotalXSec(
      XSecReduction const strategy = XSecReduction::Mean) const noexcept -> F
  {
    ASSERT(!_t.empty());
    if (strategy == XSecReduction::Max) {
      return *um2::max_element(_t.cbegin(), _t.cend());
    }
    return um2::mean(_t.cbegin(), _t.cend());
  }
}; // class XS

} // namespace um2
