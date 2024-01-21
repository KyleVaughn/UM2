#pragma once

#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/math.hpp>
#include <um2/stdlib/vector.hpp>

//==============================================================================
// ANGULAR QUADRATURE
//==============================================================================
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

namespace um2
{

enum class AngularQuadratureType { Chebyshev };

// Inputs
//  degree: The number of angles to use in the quadrature
//  weights: The vector to store the weights in
//  angles: The vector to store the angles in
//
// Preconditions
//  degree > 0
//
// Postconditions
//  weights.size() == degree
//  angles.size() == degree
//  
// The weights will all be equal to 1/degree.
// The angles will be evenly spaced in the range (0, π/2).
HOSTDEV constexpr static void
setChebyshevAngularQuadrature(Size degree, um2::Vector<F> & weights,
                              um2::Vector<F> & angles) noexcept
{
  ASSERT_ASSUME(degree > 0);
  // A Chebyshev-type quadrature for a given weight function is a quadrature formula
  // with equal weights. This function produces evenly spaced angles with equal weights.

  // Weights
  weights.resize(degree);
  F const wt = static_cast<F>(1) / static_cast<F>(degree);
  um2::fill(weights.begin(), weights.end(), wt);

  // Angles
  angles.resize(degree);
  F const pi_deg = pi_4<F> * wt;
  for (Size i = 0; i < degree; ++i) {
    angles[i] = pi_deg * static_cast<F>(2 * i + 1);
  }
}


// An angular quadrature that is the product of two 1D angular quadratures.
// Due to symmetry, both polar andazimuthal angles are only stored in the 
// range (0, π/2).
class ProductAngularQuadrature
{

  Vector<F> _wazi; // Weights for the azimuthal angles
  Vector<F> _azi;  // Azimuthal angles, γ ∈ (0, π/2)
  Vector<F> _wpol; // Weights for the polar angles
  Vector<F> _pol;  // Polar angles, θ ∈ (0, π/2)

public:
  //============================================================================
  // Constructors
  //============================================================================

  constexpr ProductAngularQuadrature() noexcept = default;

  HOSTDEV constexpr ProductAngularQuadrature(AngularQuadratureType azi_form,
                                             Size azi_degree,
                                             AngularQuadratureType pol_form,
                                             Size pol_degree) noexcept
  {
    ASSERT_ASSUME(azi_degree > 0);
    ASSERT_ASSUME(pol_degree > 0);
    switch (azi_form) {
    case AngularQuadratureType::Chebyshev:
      setChebyshevAngularQuadrature(azi_degree, _wazi, _azi);
      break;
    default:
      __builtin_unreachable();
    }

    switch (pol_form) {
    case AngularQuadratureType::Chebyshev:
      setChebyshevAngularQuadrature(pol_degree, _wpol, _pol);
      break;
    default:
      __builtin_unreachable();
    }
  }

  //============================================================================
  // Accessors
  //============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  azimuthalDegree() const noexcept -> Size
  {
    return _wazi.size();
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  polarDegree() const noexcept -> Size
  {
    return _wpol.size();
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  azimuthalWeights() const noexcept -> Vector<F> const &
  {
    return _wazi;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  azimuthalAngles() const noexcept -> Vector<F> const &
  {
    return _azi;
  }

  PURE HOSTDEV [[nodiscard]] constexpr auto
  polarWeights() const noexcept -> Vector<F> const &
  {
    return _wpol;
  }

  PURE [[nodiscard]] constexpr auto
  polarAngles() const noexcept -> Vector<F> const &
  {
    return _pol;
  }
};

} // namespace um2
