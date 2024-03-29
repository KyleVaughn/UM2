#pragma once

#include <um2/config.hpp>

#include <um2/common/Log.hpp>
#include <um2/stdlib/Vector.hpp>
#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/math.hpp>

namespace um2
{

// Angular quadrature defined on the unit sphere octant in the upper right,
// closest to the viewer. The angles and weights are transformed to the other
// octants by symmetry.
//     +----+----+
//    /    /    /|
//   +----+----+ |
//  /    /    /| +
// +----+----+ |/|
// |    |this| + |
// |    | one|/| +
// +----+----+ |/
// |    |    | +
// |    |    |/
// +----+----+
//
// The spherical coordinates are defined in the following manner
// Ω̂ = (Ω_i, Ω_j, Ω_k) = (cos(θ),  sin(θ)cos(γ),  sin(θ)sin(γ))
//                     = (     μ, √(1-μ²)cos(γ), √(1-μ²)sin(γ))
//
//        j
//        ^
//        |   θ is the polar angle about the i-axis (x-direction)
//        |   γ is the azimuthal angle in the j-k plane, from the j-axis
//        |
//        |
//        |
//       /|
//      (γ|
//       \|--------------------> i
//       / \θ)                          //
//      /   \                           //
//     /     \                          //
//    /       \ Ω̂                       //
//   /         v                        //
//  𝘷                                   //
//  k                                   //

enum class AngularQuadratureType { None, Chebyshev };

template <std::floating_point T>
HOSTDEV constexpr static void
setChebyshevAngularQuadrature(Size degree, um2::Vector<T> & weights,
                              um2::Vector<T> & angles)
{
  // A Chebyshev-type quadrature for a given weight function is a quadrature formula
  // with equal weights. This function produces evenly spaced angles with equal weights.

  // Weights
  weights.resize(degree);
  T const wt = static_cast<T>(1) / static_cast<T>(degree);
  um2::fill(weights.begin(), weights.end(), wt);

  // Angles
  angles.resize(degree);
  T const degree_4 = static_cast<T>(degree) * static_cast<T>(4);
  T const pi_deg = pi<T> / degree_4;
  for (Size i = 0; i < degree; ++i) {
    angles[i] = pi_deg * static_cast<T>(2 * i + 1);
  }
}

template <std::floating_point T>
struct ProductAngularQuadrature {
  um2::Vector<T> wazi; // Weights for the azimuthal angles
  um2::Vector<T> azi;  // Azimuthal angles, γ ∈ (0, π/2)
  um2::Vector<T> wpol; // Weights for the polar angles
  um2::Vector<T> pol;  // Polar angles, θ ∈ (0, π/2)

  //============================================================================
  // Constructors
  //============================================================================

  constexpr ProductAngularQuadrature() noexcept = default;

  HOSTDEV constexpr ProductAngularQuadrature(AngularQuadratureType azi_form,
                                             Size azi_degree,
                                             AngularQuadratureType pol_form,
                                             Size pol_degree) noexcept
  {
    switch (azi_form) {
    case AngularQuadratureType::Chebyshev:
      setChebyshevAngularQuadrature(azi_degree, wazi, azi);
      break;
    default:
      assert(false);
      break;
    }

    switch (pol_form) {
    case AngularQuadratureType::Chebyshev:
      setChebyshevAngularQuadrature(pol_degree, wpol, pol);
      break;
    default:
      assert(false);
      break;
    }
  }
};

} // namespace um2
