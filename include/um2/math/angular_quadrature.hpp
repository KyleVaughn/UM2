#pragma once

#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/math.hpp>
#include <um2/stdlib/vector.hpp>

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

enum class AngularQuadratureType { Chebyshev };

template <std::floating_point T>
HOSTDEV constexpr static void
setChebyshevAngularQuadrature(Size degree, um2::Vector<T> & weights,
                              um2::Vector<T> & angles)
{
  ASSERT_ASSUME(degree > 0);
  // A Chebyshev-type quadrature for a given weight function is a quadrature formula
  // with equal weights. This function produces evenly spaced angles with equal weights.

  // Weights
  weights.resize(degree);
  T const wt = static_cast<T>(1) / static_cast<T>(degree);
  um2::fill(weights.begin(), weights.end(), wt);

  // Angles
  angles.resize(degree);
  T const pi_deg = pi_4<T> * wt;
  for (Size i = 0; i < degree; ++i) {
    angles[i] = pi_deg * static_cast<T>(2 * i + 1);
  }
}

template <std::floating_point T>
class ProductAngularQuadrature {

  Vector<T> _wazi; // Weights for the azimuthal angles
  Vector<T> _azi;  // Azimuthal angles, γ ∈ (0, π/2)
  Vector<T> _wpol; // Weights for the polar angles
  Vector<T> _pol;  // Polar angles, θ ∈ (0, π/2)

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

  HOSTDEV [[nodiscard]] constexpr auto 
  azimuthalDegree() const noexcept -> Size
  {
    return _wazi.size();
  }

  HOSTDEV [[nodiscard]] constexpr auto
  polarDegree() const noexcept -> Size
  {
    return _wpol.size();
  }

  HOSTDEV [[nodiscard]] constexpr auto
  azimuthalWeights() const noexcept -> Vector<T> const &
  {
    return _wazi;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  azimuthalAngles() const noexcept -> Vector<T> const &
  {
    return _azi;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  polarWeights() const noexcept -> Vector<T> const &
  {
    return _wpol;
  }

  HOSTDEV [[nodiscard]] constexpr auto
  polarAngles() const noexcept -> Vector<T> const &
  {
    return _pol;
  }

};

} // namespace um2
