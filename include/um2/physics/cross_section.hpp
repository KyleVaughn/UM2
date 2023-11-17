#pragma once

#include <um2/math/stats.hpp>
#include <um2/stdlib/vector.hpp>

namespace um2
{

enum class XSReductionStrategy {
  Max,
  Mean,
};

template <std::floating_point T>
struct CrossSection {

  Vector<T> t; // Macroscopic total cross section

  //======================================================================
  // Constructors
  //======================================================================

  constexpr CrossSection() noexcept = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr CrossSection(Vector<T> const & t_in) noexcept
      : t(t_in)
  {
#if UM2_ENABLE_DBC
    ASSERT(!t.empty());
    for (auto const & t_i : t) {
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
    ASSERT(!t.empty());
    if (strategy == XSReductionStrategy::Max) {
      return *std::max_element(t.cbegin(), t.cend());
    }
    return um2::mean(t.cbegin(), t.cend());
  }
};

} // namespace um2
