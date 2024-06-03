#pragma once

#include <um2/config.hpp>

#include <um2/math/matrix.hpp>

#include <complex>

//======================================================================
// CMFD
//======================================================================
// Coarse Mesh Finite Difference (CMFD) acceleration. Currently we do
// not implement CMFD, but we wish to compute the approximate spectral
// radius of the CMFD accelerated fixed-source iteration using Fourier
// analysis.
//
// The functions in this file compute the spectral radius of a 1D, 1-group
// fixed-source iteration with CMFD acceleration based on the following
// paper:
//   Analysis of Stabilization Techniques for CMFD
//   Acceleration of Neutron Transport Problems
//   by M. Jarrett, B. Kochunas, A. Zhu, and T. Downar
//   Nuclear Science and Engineering 2016
//
// ASSUMPTIONS:
// - K = 1: A single coarse cell with periodic boundary conditions.

// NOLINTBEGIN(readability-identifier-naming)

namespace um2
{

using ComplexF = std::complex<Float>;

struct CMFDCellParams
{
  // Suspend naming conventions for this struct
  Float w;        // Width of coarse cell
  Int p;          // Number of fine cells per coarse cell
  Float Sigma_t;  // Total cross section (cm^-1)
  Float c;        // Scattering ratio
  Int s;          // Number of sweeps
  Float eta;      // Diffusion coefficient modifier η ∈ [0, 1/4]

  // Computed parameters
  Float h;     // Fine cell thickness (cm)
  Float delta; // Equation 56, optical thickness of coarse cell
  Float D;     // Equation 27, diffusion coefficient

  static Int constexpr N = 32; // Polar quadrature order, θ ∈ (0, 2π), hence μ ∈ (-1, 1)
  static Int constexpr num_alpha = 256; // Number of α values to test

  // Constructor
  inline
  CMFDCellParams(Float iw, Int ip, Float iSigma_t, Float ic, Int is, Float ieta)
    : w(iw), p(ip), Sigma_t(iSigma_t), c(ic), s(is), eta(ieta)
  {
    ASSERT(0 < iw);
    ASSERT(1 <= ip);
    ASSERT(0 < iSigma_t);
    ASSERT(0 <= ic);
    ASSERT(ic <= 1);
    ASSERT(0 < is);
    ASSERT(0 <= ieta);
    ASSERT(ieta <= 0.25);

    static_assert(N % 2 == 0);
    static_assert(num_alpha > 0);

    h = w / p;
    delta = Sigma_t * w;
    D = 1.0 / (3.0 * delta) + eta;
  }

};

} // namespace um2

namespace um2::cmfd
{


CONST auto
beta_n(Float h, Float mu_n) -> Float;

void
set_An(Matrix<ComplexF> & An, CMFDCellParams const & params, Float mu_n, Float alpha);

void
set_Bn(Matrix<ComplexF> & Bn, Matrix<ComplexF> const & An, 
    CMFDCellParams const & params, Float mu_n, Float alpha);

void
set_U(Matrix<ComplexF> & U, Matrix<ComplexF> & An, Matrix<ComplexF> & Bn,
    CMFDCellParams const & params, Float alpha, Matrix<ComplexF> & I,
    Vector<Int> & ipiv);

PURE auto
getF(CMFDCellParams const & params, Float alpha) -> Float;

void
set_omega(Matrix<ComplexF> & omega, Matrix<ComplexF> & An, Matrix<ComplexF> & Bn,
    Matrix<ComplexF> & U, Matrix<ComplexF> const & J, CMFDCellParams const & params, Float alpha,
    Matrix<ComplexF> & I, Vector<Int> & ipiv); 

} // namespace um2::cmfd


namespace um2
{

PURE auto
spectral_radius(CMFDCellParams const & params) -> ComplexF; 

auto    
spectral_radius(CMFDCellParams const & params,    
                Matrix<ComplexF> & An,    
                Matrix<ComplexF> & Bn,    
                Matrix<ComplexF> & U,    
                Matrix<ComplexF> & omega,    
                Matrix<ComplexF> & J,    
                Matrix<ComplexF> & I,    
                Vector<Int> & ipiv    
    ) -> ComplexF;

} // namespace um2

// NOLINTEND(readability-identifier-naming)
