#pragma once

#include <um2/math/stats.hpp>
#include <um2/stdlib/vector.hpp>

//======================================================================
// CROSS SECTION
//======================================================================
// A multi-group cross section. Currently, we only need the total cross
// section, but will add scattering, etc. later. 
//
// Multigroup cross sections are frequently reduced to a one-group cross
// section for use with the Knudsen number. Hence we provide a
// getOneGroupTotalXS() method that accepts an XSReductionStrategy, which
// will take either the mean or max of the total cross section.

namespace um2
{

enum class XSReductionStrategy {
  Max,
  Mean,
};

template <std::floating_point T>
class CrossSection
{

  Vector<T> _t; // Total macroscopic cross section

public:
  //======================================================================
  // Constructors
  //======================================================================

  constexpr CrossSection() noexcept = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr CrossSection(Vector<T> const & t) noexcept
      : _t(t)
  {
#if UM2_ENABLE_ASSERTS
    ASSERT(!_t.empty());
    for (auto const & t_i : _t) {
      ASSERT(t_i >= 0);
    }
#endif
  }

  //======================================================================
  // Methods
  //======================================================================

  [[nodiscard]] auto constexpr getOneGroupTotalXS(
      XSReductionStrategy const strategy = XSReductionStrategy::Mean) const noexcept -> T
  {
    ASSERT(!_t.empty());
    if (strategy == XSReductionStrategy::Max) {
      return *um2::max_element(_t.cbegin(), _t.cend());
    }
    return um2::mean(_t.cbegin(), _t.cend());
  }
};

} // namespace um2
