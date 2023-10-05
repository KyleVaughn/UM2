#pragma once

#include <um2/stdlib/Vector.hpp>

namespace um2
{

struct CrossSection {

  Vector<Float> t; // Macroscopic total cross section

  // ---------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------

  constexpr CrossSection() noexcept = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr CrossSection(Vector<Float> const & t_in) noexcept : t(t_in) {}

};

} // namespace um2
