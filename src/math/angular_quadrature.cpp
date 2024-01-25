#include <um2/math/angular_quadrature.hpp>

namespace um2
{

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
HOSTDEV static void
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

//============================================================================
// Constructors
//============================================================================


HOSTDEV 
ProductAngularQuadrature::ProductAngularQuadrature(AngularQuadratureType azi_form,
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

} // namespace um2
