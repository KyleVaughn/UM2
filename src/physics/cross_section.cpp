#include <um2/physics/cross_section.hpp>

#include <um2/common/logger.hpp>
#include <um2/math/stats.hpp>

namespace um2
{

//==============================================================================
// Member functions
//==============================================================================

void
XSec::validate() const noexcept
{
  if (_num_groups <= 0) {
    LOG_ERROR("Cross section has a non-positive number of groups");
  }

  if (_a.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of absorption values");
  }

  if (_f.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of fission values");
  }

  if (_nuf.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of nu*fission values"); 
  }

  if (_tr.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of transport values");
  }

  if (_s.size() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect number of scattering values");
  }

  if (_ss.rows() != _num_groups || _ss.cols() != _num_groups) {
    LOG_ERROR("Cross section has an incorrect scattering matrix size"); 
  }

  // If the cross section is macroscopic, the total XS must be positive. 
  // If the cross section is microscopic, the total XS can be negative due to
  // correction factors.
  if (isMacro()) {
    for (auto const & a : _a) {
      if (a < 0.0) {
        LOG_ERROR("Cross section has a negative absorption value"); 
      }
    }

    for (auto const & f : _f) {
      if (f < 0.0) {
        LOG_ERROR("Cross section has a negative fission value");
      }
    }

    for (auto const & nuf : _nuf) {
      if (nuf < 0.0) {
        LOG_ERROR("Cross section has a negative nu*fission value");
      }
    }

    for (auto const & tr : _tr) {
      if (tr < 0.0) {
        LOG_ERROR("Cross section has a negative transport value");
      }
    }

    for (auto const & s : _s) {
      if (s < 0.0) {
        LOG_ERROR("Cross section has a negative scattering value");
      }
    }
    for (Int i = 0; i < _num_groups * _num_groups; ++i) {
      if (_ss(i) < 0.0) {
        LOG_ERROR("Cross section has a negative scattering matrix value");
      }
    }
  }
}

auto
XSec::collapseTo1GroupAvg() const noexcept -> XSec
{
  XSec result(1);
  result.isMacro() = isMacro();
  ASSERT(_num_groups > 0);
  result.a()[0] = um2::mean(_a.cbegin(), _a.cend());
  result.f()[0] = um2::mean(_f.cbegin(), _f.cend());
  result.nuf()[0] = um2::mean(_nuf.cbegin(), _nuf.cend());
  result.tr()[0] = um2::mean(_tr.cbegin(), _tr.cend());
  result.s()[0] = um2::mean(_s.cbegin(), _s.cend());
  result.ss()(0) = _num_groups * um2::mean(_ss.begin(), _ss.end());
  auto constexpr eps = 1.0e-6;
  ASSERT_NEAR(result.s()[0], result.ss()(0), eps); 
  result.validate();
  return result;
}

    
PURE [[nodiscard]] auto    
getC5G7XSecs() noexcept -> Vector<XSec>
{
  // UO2
  //         Total       Transport   Absorption  Capture     Fission     Nu          Chi 
  // Group 1 2.12450E-01 1.77949E-01 8.02480E-03 8.12740E-04 7.21206E-03 2.78145E+00 5.87910E-01 
  // Group 2 3.55470E-01 3.29805E-01 3.71740E-03 2.89810E-03 8.19301E-04 2.47443E+00 4.11760E-01 
  // Group 3 4.85540E-01 4.80388E-01 2.67690E-02 2.03158E-02 6.45320E-03 2.43383E+00 3.39060E-04 
  // Group 4 5.59400E-01 5.54367E-01 9.62360E-02 7.76712E-02 1.85648E-02 2.43380E+00 1.17610E-07 
  // Group 5 3.18030E-01 3.11801E-01 3.00200E-02 1.22116E-02 1.78084E-02 2.43380E+00 0.00000E+00 
  // Group 6 4.01460E-01 3.95168E-01 1.11260E-01 2.82252E-02 8.30348E-02 2.43380E+00 0.00000E+00 
  // Group 7 5.70610E-01 5.64406E-01 2.82780E-01 6.67760E-02 2.16004E-01 2.43380E+00 0.00000E+00
  // 
  // Scatter Matrix
  //         to Group 1  to Group 2  to Group 3  to Group 4  to Group 5  to Group 6  to Group 7 
  // Group 1 1.27537E-01 4.23780E-02 9.43740E-06 5.51630E-09 0.00000E+00 0.00000E+00 0.00000E+00 
  // Group 2 0.00000E+00 3.24456E-01 1.63140E-03 3.14270E-09 0.00000E+00 0.00000E+00 0.00000E+00 
  // Group 3 0.00000E+00 0.00000E+00 4.50940E-01 2.67920E-03 0.00000E+00 0.00000E+00 0.00000E+00 
  // Group 4 0.00000E+00 0.00000E+00 0.00000E+00 4.52565E-01 5.56640E-03 0.00000E+00 0.00000E+00 
  // Group 5 0.00000E+00 0.00000E+00 0.00000E+00 1.25250E-04 2.71401E-01 1.02550E-02 1.00210E-08 
  // Group 6 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 1.29680E-03 2.65802E-01 1.68090E-02 
  // Group 7 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 8.54580E-03 2.73080E-01
  //
  // Note: we store scattering matrices ss(to group, from group) in colume major order. Hence, the
  // 1D memory layout is equivalent to the 2D memory row-major layout in the table above.
  Int constexpr num_groups = 7;
  XSec uo2(num_groups);
  uo2.a() = {8.02480e-03, 3.71740e-03, 2.67690e-02, 9.62360e-02, 3.00200e-02, 1.11260e-01, 2.82780e-01};
  uo2.f() = {7.21206e-03, 8.19301e-04, 6.45320e-03, 1.85648e-02, 1.78084e-02, 8.30348e-02, 2.16004e-01};
  Vector<Float> nu = {2.78145, 2.47443, 2.43383, 2.43380, 2.43380, 2.43380, 2.43380};
  for (Int i = 0; i < num_groups; ++i) {
    uo2.nuf()[i] = nu[i] * uo2.f()[i];
  }
  uo2.tr() = {1.77949e-01, 3.29805e-01, 4.80388e-01, 5.54367e-01, 3.11801e-01, 3.95168e-01, 5.64406e-01};
  // clang-format off
  uo2.ss().asVector() = {
    1.27537e-01, 4.23780e-02, 9.43740e-06, 5.51630e-09, 0.00000e+00, 0.00000e+00, 0.00000e+00, 
    0.00000e+00, 3.24456e-01, 1.63140e-03, 3.14270e-09, 0.00000e+00, 0.00000e+00, 0.00000e+00, 
    0.00000e+00, 0.00000e+00, 4.50940e-01, 2.67920e-03, 0.00000e+00, 0.00000e+00, 0.00000e+00, 
    0.00000e+00, 0.00000e+00, 0.00000e+00, 4.52565e-01, 5.56640e-03, 0.00000e+00, 0.00000e+00, 
    0.00000e+00, 0.00000e+00, 0.00000e+00, 1.25250e-04, 2.71401e-01, 1.02550e-02, 1.00210e-08, 
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.29680e-03, 2.65802e-01, 1.68090e-02, 
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.54580e-03, 2.73080e-01};
  // clang-format on
  // Sum each column to get the total scattering cross section
  for (Int i = 0; i < num_groups; ++i) {
    // uo2.s()[i] = 0.0; should already be zero initialized
    for (Int j = 0; j < num_groups; ++j) {
      uo2.s()[i] += uo2.ss()(j, i);
    }
  }
  uo2.validate();

  return {uo2};
}

} // namespace um2
