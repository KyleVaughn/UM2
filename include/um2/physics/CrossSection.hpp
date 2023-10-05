#pragma once

#include <um2/stdlib/Vector.hpp>

namespace um2
{

template <typename T>
struct CrossSection {

  Vector<T> t; // Macroscopic total cross section

  // ---------------------------------------------------------------------
  // Constructors
  // ---------------------------------------------------------------------

  constexpr CrossSection() noexcept = default;

};

} // namespace um2
