#include <um2/physics/cmfd.hpp>

#if UM2_HAS_CMFD
#  include <um2/config.hpp>
#  include <um2/math/matrix.hpp>
#  include <um2/stdlib/algorithm/fill.hpp>
#  include <um2/stdlib/assert.hpp>
#  include <um2/stdlib/math/abs.hpp>
#  include <um2/stdlib/math/exponential_functions.hpp>
#  include <um2/stdlib/math/hyperbolic_functions.hpp>
#  include <um2/stdlib/math/trigonometric_functions.hpp>
#  include <um2/stdlib/numbers.hpp>
#  include <um2/stdlib/vector.hpp>

#  include <complex>

// TODO(kcvaughn): Clean up all the unnecessary memory allocations

namespace um2::cmfd
{

// NOLINTBEGIN(readability-identifier-naming)

// Equation 5. No memory allocation
CONST auto
beta_n(Float const h, Float const mu_n) -> Float
{
  Float const z = h / (2 * mu_n);
  return 1 / um2::tanh(z) - 1 / z;
}

// Equation 51. No memory allocation
void
set_An(Matrix<ComplexF> & An, CMFDCellParams const & params, Float const mu_n,
       Float const alpha)
{
  // ASSUMPTIONS:
  // - An is zero everywhere but the diagonal, superdiagonal, and (p-1, 0)

  ASSERT(An.rows() == params.p);
  ASSERT(An.cols() == params.p);

  // Let d = (1 - βₙ) / 2 be the value on the diagonal
  // Let u = (1 + βₙ) / 2 be the value on the upper diagonal
  // The lower left corner of the matrix is u * exp(i α Δ)
  Float const beta = beta_n(params.h, mu_n);
  ComplexF const d((1 - beta) / 2, 0);
  ComplexF const u((1 + beta) / 2, 0);

  // Traverse in column-major order
  Int const p = params.p;
  ComplexF const im(0, 1);
  An(0, 0) = d;
  An(p - 1, 0) = u * um2::exp(im * alpha * params.delta);
  for (Int i = 1; i < p; ++i) {
    An(i - 1, i) = u;
    An(i, i) = d;
  }
}

// Equation 52. No memory allocation
void
set_Bn(Matrix<ComplexF> & Bn, Matrix<ComplexF> const & An, CMFDCellParams const & params,
       Float const mu_n, Float const alpha)
{
  // ASSUMPTIONS:
  // - Bn is zero everywhere but the diagonal, superdiagonal, and (p-1, 0)
  // - An is already set

  ASSERT(Bn.rows() == params.p);
  ASSERT(Bn.cols() == params.p);

  // Let d = -μₙ / (Σ_t * h) be the value on the diagonal
  // Let u = μₙ / (Σ_t * h) be the value on the upper diagonal
  // The lower left corner of the matrix is u * exp(i α Δ)
  // Note: d = -u
  ComplexF const u(mu_n / (params.Sigma_t * params.h), 0);
  ComplexF const d = -u;

  // Traverse in column-major order
  Int const p = params.p;
  ComplexF const im(0, 1);
  Bn(0, 0) = d;
  Bn(p - 1, 0) = u * um2::exp(im * alpha * params.delta);
  for (Int i = 1; i < p; ++i) {
    Bn(i - 1, i) = u;
    Bn(i, i) = d;
  }
  Bn += An;
}

// Equation 54. No memory allocation
void
set_U(Matrix<ComplexF> & U, Matrix<ComplexF> & An, Matrix<ComplexF> & Bn,
      CMFDCellParams const & params, Float const alpha, Matrix<ComplexF> & I,
      Vector<Int> & ipiv)
{
  // ASSUMPTIONS:
  // - An, Bn, and I are p by p matrices

  ASSERT(U.rows() == params.p);
  ASSERT(U.cols() == params.p);
  ASSERT(An.rows() == params.p);
  ASSERT(An.cols() == params.p);
  ASSERT(Bn.rows() == params.p);
  ASSERT(Bn.cols() == params.p);
  ASSERT(I.rows() == params.p);
  ASSERT(I.cols() == params.p);

  // Zero out U
  U.zero();

  // We don't currently have the ability to compute Gauss-Legendre quadrature
  // at runtime, so we will hardcode the values for N = 32

  ASSERT(params.N == 32);
  Float constexpr mu[32] = {
      -0.9972638618494816,   -0.9856115115452684,  -0.9647622555875064,
      -0.9349060759377397,   -0.8963211557660521,  -0.84936761373257,
      -0.7944837959679424,   -0.7321821187402897,  -0.6630442669302152,
      -0.5877157572407623,   -0.5068999089322294,  -0.42135127613063533,
      -0.33186860228212767,  -0.23928736225213706, -0.1444719615827965,
      -0.048307665687738324, 0.048307665687738324, 0.1444719615827965,
      0.23928736225213706,   0.33186860228212767,  0.42135127613063533,
      0.5068999089322294,    0.5877157572407623,   0.6630442669302152,
      0.7321821187402897,    0.7944837959679424,   0.84936761373257,
      0.8963211557660521,    0.9349060759377397,   0.9647622555875064,
      0.9856115115452684,    0.9972638618494816};
  Float constexpr w[32] = {
      0.007018610009470092, 0.016274394730905643, 0.025392065309262066,
      0.03427386291302141,  0.04283589802222672,  0.05099805926237615,
      0.05868409347853548,  0.0658222227763618,   0.07234579410884856,
      0.07819389578707032,  0.08331192422694678,  0.08765209300440377,
      0.09117387869576384,  0.09384439908080465,  0.09563872007927489,
      0.09654008851472778,  0.09654008851472778,  0.09563872007927489,
      0.09384439908080465,  0.09117387869576384,  0.08765209300440377,
      0.08331192422694678,  0.07819389578707032,  0.07234579410884856,
      0.0658222227763618,   0.05868409347853548,  0.05099805926237615,
      0.04283589802222672,  0.03427386291302141,  0.025392065309262066,
      0.016274394730905643, 0.007018610009470092};

  // Loop over the polar quadrature to compute equation 54
  // U = (c / 2) ∑_n^N wₙ * Aₙ * (Bₙ)^-1

  // Find X such that Bn * X = I
  Int const p = params.p;
  for (Int n = 0; n < um2::CMFDCellParams::N; ++n) {
    I.zero();
    for (Int i = 0; i < p; ++i) {
      I(i, i) = ComplexF(1, 0);
    }
    set_An(An, params, mu[n], alpha);
    Bn.zero();
    set_Bn(Bn, An, params, mu[n], alpha);
    // auto const B_inv = linearSolve(Bn, I);
    // auto const X = An * B_inv;
    linearSolve(Bn, I, ipiv);
    // I is now Bn⁻¹
    auto const & B_inv = I;
    matmul(Bn, An, B_inv);
    auto const & X = Bn;
    auto const w_n = w[n];
    // U = U + wₙ * X
    for (Int i = 0; i < p * p; ++i) {
      U(i) += w_n * X(i);
    }
  }
  U *= params.c / 2;
}

// Equation 56. No memory allocation
PURE auto
getF(CMFDCellParams const & params, Float alpha) -> Float
{
  Float const c = params.c;
  Float const delta = params.delta;
  Float const numerator = c * params.Sigma_t * params.h;
  Float const denominator_l = 2 * params.D * (1 - um2::cos(alpha * delta));
  Float const denominator_r = (1 - c) * delta;
  return numerator / (denominator_l + denominator_r);
}

// Equation 57 (just constructing the matrix). No memory allocation
void
set_omega(Matrix<ComplexF> & omega, Matrix<ComplexF> & An, Matrix<ComplexF> & Bn,
          Matrix<ComplexF> & U, Matrix<ComplexF> const & J, CMFDCellParams const & params,
          Float alpha, Matrix<ComplexF> & I, Vector<Int> & ipiv)
{
  // ASSUMPTIONS:
  //  - Aₙ and Bₙ are zero everywhere except the locations we will set
  //  - J is a matrix of ones
  Int const p = params.p;
  set_U(U, An, Bn, params, alpha, I, ipiv);
  auto const F = getF(params, alpha); // This is a scalar
  auto const s = params.s;
  ASSERT(s == 1);

  // ω = (U^σ + F * J * (U^σ - U^(σ - 1)))
  // ω = (U + F * J * (U - I)) * U^(σ - 1)

  ComplexF const one(1, 0);
  // U' = U - I
  for (Int i = 0; i < p; ++i) {
    U(i, i) -= one;
  }
  // ω = J * U' = (J * (U - I))
  matmul(omega, J, U);
  // ω = F * (J * (U - I))
  omega *= F;
  // ω = (U - I) + F * (J * (U - I))
  omega += U;
  // ω = U + F * (J * (U - I))
  for (Int i = 0; i < p; ++i) {
    omega(i, i) += one;
  }

  // if (s > 1) {
  //   Matrix<ComplexF> U_s1 = U; // U^1
  //   // Don't have M^pow yet
  //   for (Int i = 0; i < s - 2; ++i) {
  //     U_s1 = U_s1 * U;
  //   }
  //   omega = omega * U_s1;
  // }
}

} // namespace um2::cmfd

namespace um2
{

PURE auto
spectral_radius(CMFDCellParams const & params) -> ComplexF
{

  // Set up α samples. We sample α ∈ [0, π/Δ] (Equation 58)
  Int const num_alpha = CMFDCellParams::num_alpha;
  Float const alpha_max = um2::pi<Float> / params.delta;
  Float const d_alpha = alpha_max / (CMFDCellParams::num_alpha - 1);

  // Set up matrices
  Int const p = params.p;
  // An and Bn must be initialized to zero
  Matrix<ComplexF> An(p, p);
  Matrix<ComplexF> Bn(p, p);
  An.zero();
  Bn.zero();

  // U and omega don't need to be initialized
  Matrix<ComplexF> U(p, p);
  Matrix<ComplexF> omega(p, p);

  // J is a matrix of ones
  Matrix<ComplexF> J(p, p);
  um2::fill(J.begin(), J.end(), ComplexF(1.0, 0.0));

  // These don't need to be initialized
  Matrix<ComplexF> I(p, p); // Identity matrix
  Vector<Int> ipiv(p);

  ComplexF r(0, 0);
  Float r_abs = 0;
  for (Int i = 0; i < num_alpha; ++i) {
    auto const a = i * d_alpha;
    cmfd::set_omega(omega, An, Bn, U, J, params, a, I, ipiv);
    auto const eigs = eigvals(omega);
    // get the maximum eigenvalue
    for (auto const & eig : eigs) {
      Float const eig_abs = um2::abs(eig);
      if (eig_abs > r_abs) {
        r = eig;
        r_abs = eig_abs;
      }
    }
  }
  return r;
}

auto
spectral_radius(CMFDCellParams const & params, Matrix<ComplexF> & An,
                Matrix<ComplexF> & Bn, Matrix<ComplexF> & U, Matrix<ComplexF> & omega,
                Matrix<ComplexF> & J, Matrix<ComplexF> & I,
                Vector<Int> & ipiv) -> ComplexF
{

  // Set up α samples. We sample α ∈ [0, π/Δ] (Equation 58)
  Int const num_alpha = CMFDCellParams::num_alpha;
  Float const alpha_max = um2::pi<Float> / params.delta;
  Float const d_alpha = alpha_max / (CMFDCellParams::num_alpha - 1);

  // Set up matrices
  An.zero();
  Bn.zero();

  um2::fill(J.begin(), J.end(), ComplexF(1.0, 0.0));

  ComplexF r(0, 0);
  Float r_abs = 0;
  for (Int i = 0; i < num_alpha; ++i) {
    auto const a = i * d_alpha;
    cmfd::set_omega(omega, An, Bn, U, J, params, a, I, ipiv);
    auto const eigs = eigvals(omega);
    // get the maximum eigenvalue
    for (auto const & eig : eigs) {
      Float const eig_abs = um2::abs(eig);
      if (eig_abs > r_abs) {
        r = eig;
        r_abs = eig_abs;
      }
    }
  }
  return r;
}

} // namespace um2

// NOLINTEND(readability-identifier-naming)
#endif // UM2_HAS_CMFD
