#include <um2/config.hpp>
#include <um2/physics/cross_section.hpp>

#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/vector.hpp>

#include <um2/common/logger.hpp>
#include <um2/math/stats.hpp>

namespace um2
{

//==============================================================================
// Member functions
//==============================================================================

void
// NOLINTNEXTLINE(*cognitive*)
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

  if (isMacro()) {
    for (Int i = 0; i < _num_groups; ++i) {
      auto const a = _a[i];
      if (a < 0.0) {
        LOG_WARN("Cross section has a negative absorption cross section in group ", i,
                 " (", a, ")");
      }
    }

    for (Int i = 0; i < _num_groups; ++i) {
      auto const f = _f[i];
      if (f < 0.0) {
        LOG_WARN("Cross section has a negative fission cross section in group ", i, " (",
                 f, ")");
      }
      if (f > 0.0 && !isFissile()) {
        LOG_WARN("Cross section has a positive fission cross section in group ", i,
                 " but is not fissile");
      }
    }

    for (Int i = 0; i < _num_groups; ++i) {
      auto const nuf = _nuf[i];
      if (nuf < 0.0) {
        LOG_WARN("Cross section has a negative nu*fission cross section in group ", i,
                 " (", nuf, ")");
      }
      if (nuf > 0.0 && !isFissile()) {
        LOG_WARN("Cross section has a positive nu*fission cross section in group ", i,
                 " but is not fissile");
      }
    }

    for (Int i = 0; i < _num_groups; ++i) {
      auto const tr = _tr[i];
      if (tr < 0.0) {
        LOG_WARN("Cross section has a negative transport cross section in group ", i,
                 " (", tr, ")");
      }
    }

    for (Int i = 0; i < _num_groups; ++i) {
      auto const s = _s[i];
      if (s < 0.0) {
        LOG_WARN("Cross section has a negative scattering cross section in group ", i,
                 " (", s, ")");
      }
    }
    for (Int i = 0; i < _num_groups * _num_groups; ++i) {
      if (_ss(i) < 0.0) {
        LOG_WARN("Cross section has a negative scattering matrix value at index ", i,
                 " (", _ss(i), ")");
      }
    }
  } else {
    for (Int i = 0; i < _num_groups; ++i) {
      auto const f = _f[i];
      if (f > 0.0 && !isFissile()) {
        LOG_ERROR("Cross section has a positive fission cross section in group ", i,
                  " but is not fissile");
      }
    }

    for (Int i = 0; i < _num_groups; ++i) {
      auto const nuf = _nuf[i];
      if (nuf > 0.0 && !isFissile()) {
        LOG_ERROR("Cross section has a positive nu*fission cross section in group ", i,
                  " but is not fissile");
      }
    }
  }
}

auto
XSec::collapseTo1GroupAvg(Vector<Float> const & weights) const noexcept -> XSec
{
  XSec result(1);
  result.isMacro() = isMacro();
  result.isFissile() = isFissile();
  ASSERT(_num_groups > 0);
  if (weights.empty()) {
    result.a()[0] = um2::mean(_a.cbegin(), _a.cend());
    result.f()[0] = um2::mean(_f.cbegin(), _f.cend());
    result.nuf()[0] = um2::mean(_nuf.cbegin(), _nuf.cend());
    result.tr()[0] = um2::mean(_tr.cbegin(), _tr.cend());
    result.s()[0] = um2::mean(_s.cbegin(), _s.cend());
    result.ss()(0) = _num_groups * um2::mean(_ss.begin(), _ss.end());
  } else {
    ASSERT(weights.size() == _num_groups);
    result.a()[0] = 0;
    result.f()[0] = 0;
    result.nuf()[0] = 0;
    result.tr()[0] = 0;
    result.s()[0] = 0;
    result.ss()(0) = 0;
    for (Int i = 0; i < _num_groups; ++i) {
      result.a()[0] += weights[i] * _a[i];
      result.f()[0] += weights[i] * _f[i];
      result.nuf()[0] += weights[i] * _nuf[i];
      result.tr()[0] += weights[i] * _tr[i];
      result.s()[0] += weights[i] * _s[i];
    }
    result.ss()(0) = result.s()[0];
  }
#if UM2_ENABLE_ASSERTS
  auto constexpr eps = 1.0e-6;
  ASSERT_NEAR(result.s()[0], result.ss()(0), eps);
#endif
  result.validate();
  return result;
}

PURE [[nodiscard]] auto
// NOLINTNEXTLINE(*cognitive*)
getC5G7XSecs() noexcept -> Vector<XSec>
{
  // UO2
  //         Total       Transport   Absorption  Capture     Fission     Nu          Chi
  // Group
  // 1 2.12450E-01 1.77949E-01 8.02480E-03 8.12740E-04 7.21206E-03 2.78145E+00 5.87910E-01
  // Group
  // 2 3.55470E-01 3.29805E-01 3.71740E-03 2.89810E-03 8.19301E-04 2.47443E+00 4.11760E-01
  // Group
  // 3 4.85540E-01 4.80388E-01 2.67690E-02 2.03158E-02 6.45320E-03 2.43383E+00 3.39060E-04
  // Group
  // 4 5.59400E-01 5.54367E-01 9.62360E-02 7.76712E-02 1.85648E-02 2.43380E+00 1.17610E-07
  // Group 5 3.18030E-01 3.11801E-01 3.00200E-02 1.22116E-02 1.78084E-02 2.43380E+00
  // 0.00000E+00 Group
  // 6 4.01460E-01 3.95168E-01 1.11260E-01 2.82252E-02 8.30348E-02 2.43380E+00 0.00000E+00
  // Group 7 5.70610E-01 5.64406E-01 2.82780E-01 6.67760E-02 2.16004E-01 2.43380E+00
  // 0.00000E+00
  //
  // Scatter Matrix
  //         to Group 1  to Group 2  to Group 3  to Group 4  to Group 5  to Group 6  to
  //         Group 7
  // Group 1 1.27537E-01 4.23780E-02 9.43740E-06 5.51630E-09 0.00000E+00 0.00000E+00
  // 0.00000E+00 Group 2 0.00000E+00 3.24456E-01 1.63140E-03 3.14270E-09 0.00000E+00
  // 0.00000E+00 0.00000E+00 Group 3 0.00000E+00 0.00000E+00 4.50940E-01 2.67920E-03
  // 0.00000E+00 0.00000E+00 0.00000E+00 Group 4 0.00000E+00 0.00000E+00
  // 0.00000E+00 4.52565E-01 5.56640E-03 0.00000E+00 0.00000E+00 Group 5 0.00000E+00
  // 0.00000E+00 0.00000E+00 1.25250E-04 2.71401E-01 1.02550E-02 1.00210E-08 Group 6
  // 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 1.29680E-03 2.65802E-01 1.68090E-02
  // Group 7 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
  // 0.00000E+00 8.54580E-03 2.73080E-01
  //
  // Note: we store scattering matrices ss(to group, from group) in colume major order.
  // Hence, the 1D memory layout is equivalent to the 2D memory row-major layout in the
  // table above.
  Int constexpr num_groups = 7;
  XSec uo2(num_groups);
  uo2.isMacro() = true;
  uo2.isFissile() = true;
  uo2.a() = {8.02480e-03, 3.71740e-03, 2.67690e-02, 9.62360e-02,
             3.00200e-02, 1.11260e-01, 2.82780e-01};
  uo2.f() = {7.21206e-03, 8.19301e-04, 6.45320e-03, 1.85648e-02,
             1.78084e-02, 8.30348e-02, 2.16004e-01};
  Vector<Float> nu = {2.78145, 2.47443, 2.43383, 2.43380, 2.43380, 2.43380, 2.43380};
  for (Int i = 0; i < num_groups; ++i) {
    uo2.nuf()[i] = nu[i] * uo2.f()[i];
  }
  // NOLINTBEGIN(*use-std-numbers)
  uo2.tr() = {1.77949e-01, 3.29805e-01, 4.80388e-01, 5.54367e-01,
              3.11801e-01, 3.95168e-01, 5.64406e-01};
  // NOLINTEND(*use-std-numbers)
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

  // MOX 4.3%
  //         Total       Transport   Absorption  Capture     Fission     Nu          Chi
  // Group
  // 1 2.11920E-01 1.78731E-01 8.43390E-03 8.06860E-04 7.62704E-03 2.85209E+00 5.87910E-01
  // Group
  // 2 3.55810E-01 3.30849E-01 3.75770E-03 2.88080E-03 8.76898E-04 2.89099E+00 4.11760E-01
  // Group
  // 3 4.88900E-01 4.83772E-01 2.79700E-02 2.22717E-02 5.69835E-03 2.85486E+00 3.39060E-04
  // Group
  // 4 5.71940E-01 5.66922E-01 1.04210E-01 8.13228E-02 2.28872E-02 2.86073E+00 1.17610E-07
  // Group 5 4.32390E-01 4.26227E-01 1.39940E-01 1.29177E-01 1.07635E-02 2.85447E+00
  // 0.00000E+00 Group
  // 6 6.84950E-01 6.78997E-01 4.09180E-01 1.76423E-01 2.32757E-01 2.86415E+00 0.00000E+00
  // Group 7 6.88910E-01 6.82852E-01 4.09350E-01 1.60382E-01 2.48968E-01 2.86780E+00
  // 0.00000E+00
  //
  // Scatter Matrix
  //         to Group 1  to Group 2  to Group 3  to Group 4  to Group 5  to Group 6  to
  //         Group 7
  // Group 1 1.28876E-01 4.14130E-02 8.22900E-06 5.04050E-09 0.00000E+00 0.00000E+00
  // 0.00000E+00 Group 2 0.00000E+00 3.25452E-01 1.63950E-03 1.59820E-09 0.00000E+00
  // 0.00000E+00 0.00000E+00 Group 3 0.00000E+00 0.00000E+00 4.53188E-01 2.61420E-03
  // 0.00000E+00 0.00000E+00 0.00000E+00 Group 4 0.00000E+00 0.00000E+00
  // 0.00000E+00 4.57173E-01 5.53940E-03 0.00000E+00 0.00000E+00 Group 5 0.00000E+00
  // 0.00000E+00 0.00000E+00 1.60460E-04 2.76814E-01 9.31270E-03 9.16560E-09 Group 6
  // 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 2.00510E-03 2.52962E-01 1.48500E-02
  // Group 7 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
  // 0.00000E+00 8.49480E-03 2.65007E-01

  XSec mox43(num_groups);
  mox43.isMacro() = true;
  mox43.isFissile() = true;
  mox43.a() = {8.43390e-03, 3.75770e-03, 2.79700e-02, 1.04210e-01,
               1.39940e-01, 4.09180e-01, 4.09350e-01};
  mox43.f() = {7.62704e-03, 8.76898e-04, 5.69835e-03, 2.28872e-02,
               1.07635e-02, 2.32757e-01, 2.48968e-01};
  nu = {2.85209, 2.89099, 2.85486, 2.86073, 2.85447, 2.86415, 2.86780};
  for (Int i = 0; i < num_groups; ++i) {
    mox43.nuf()[i] = nu[i] * mox43.f()[i];
  }
  mox43.tr() = {1.78731e-01, 3.30849e-01, 4.83772e-01, 5.66922e-01,
                4.26227e-01, 6.78997e-01, 6.82852e-01};
  // clang-format off
  mox43.ss().asVector() = {
    1.28876e-01, 4.14130e-02, 8.22900e-06, 5.04050e-09, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 3.25452e-01, 1.63950e-03, 1.59820e-09, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 4.53188e-01, 2.61420e-03, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 4.57173e-01, 5.53940e-03, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 1.60460e-04, 2.76814e-01, 9.31270e-03, 9.16560e-09,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.00510e-03, 2.52962e-01, 1.48500e-02,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.49480e-03, 2.65007e-01};
  // clang-format on
  for (Int i = 0; i < num_groups; ++i) {
    for (Int j = 0; j < num_groups; ++j) {
      mox43.s()[i] += mox43.ss()(j, i);
    }
  }
  mox43.validate();

  // MOX 7.0%
  //         Total       Transport   Absorption  Capture     Fission     Nu          Chi
  // Group
  // 1 2.14540E-01 1.81323E-01 9.06570E-03 8.11240E-04 8.25446E-03 2.88498E+00 5.87910E-01
  // Group
  // 2 3.59350E-01 3.34368E-01 4.29670E-03 2.97105E-03 1.32565E-03 2.91079E+00 4.11760E-01
  // Group
  // 3 4.98910E-01 4.93785E-01 3.28810E-02 2.44594E-02 8.42156E-03 2.86574E+00 3.39060E-04
  // Group
  // 4 5.96220E-01 5.91216E-01 1.22030E-01 8.91570E-02 3.28730E-02 2.87063E+00 1.17610E-07
  // Group 5 4.80350E-01 4.74198E-01 1.82980E-01 1.67016E-01 1.59636E-02 2.86714E+00
  // 0.00000E+00 Group
  // 6 8.39360E-01 8.33601E-01 5.68460E-01 2.44666E-01 3.23794E-01 2.86658E+00 0.00000E+00
  // Group 7 8.59480E-01 8.53603E-01 5.85210E-01 2.22407E-01 3.62803E-01 2.87539E+00
  // 0.00000E+00
  //
  // Scatter Matrix
  //        to Group 1  to Group 2  to Group 3  to Group 4  to Group 5  to Group 6  to
  //        Group 7
  // Group 1 1.30457E-01 4.17920E-02 8.51050E-06 5.13290E-09 0.00000E+00 0.00000E+00
  // 0.00000E+00 Group 2 0.00000E+00 3.28428E-01 1.64360E-03 2.20170E-09 0.00000E+00
  // 0.00000E+00 0.00000E+00 Group 3 0.00000E+00 0.00000E+00 4.58371E-01 2.53310E-03
  // 0.00000E+00 0.00000E+00 0.00000E+00 Group 4 0.00000E+00 0.00000E+00
  // 0.00000E+00 4.63709E-01 5.47660E-03 0.00000E+00 0.00000E+00 Group 5 0.00000E+00
  // 0.00000E+00 0.00000E+00 1.76190E-04 2.82313E-01 8.72890E-03 9.00160E-09 Group 6
  // 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 2.27600E-03 2.49751E-01 1.31140E-02
  // Group 7 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
  // 0.00000E+00 8.86450E-03 2.59529E-01

  XSec mox70(num_groups);
  mox70.isMacro() = true;
  mox70.isFissile() = true;
  mox70.a() = {9.06570e-03, 4.29670e-03, 3.28810e-02, 1.22030e-01,
               1.82980e-01, 5.68460e-01, 5.85210e-01};
  mox70.f() = {8.25446e-03, 1.32565e-03, 8.42156e-03, 3.28730e-02,
               1.59636e-02, 3.23794e-01, 3.62803e-01};
  nu = {2.88498, 2.91079, 2.86574, 2.87063, 2.86714, 2.86658, 2.87539};
  for (Int i = 0; i < num_groups; ++i) {
    mox70.nuf()[i] = nu[i] * mox70.f()[i];
  }
  mox70.tr() = {1.81323e-01, 3.34368e-01, 4.93785e-01, 5.91216e-01,
                4.74198e-01, 8.33601e-01, 8.53603e-01};
  // clang-format off
  mox70.ss().asVector() = {
    1.30457e-01, 4.17920e-02, 8.51050e-06, 5.13290e-09, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 3.28428e-01, 1.64360e-03, 2.20170e-09, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 4.58371e-01, 2.53310e-03, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 4.63709e-01, 5.47660e-03, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 1.76190e-04, 2.82313e-01, 8.72890e-03, 9.00160e-09,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.27600e-03, 2.49751e-01, 1.31140e-02,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.86450e-03, 2.59529e-01};
  // clang-format on
  for (Int i = 0; i < num_groups; ++i) {
    for (Int j = 0; j < num_groups; ++j) {
      mox70.s()[i] += mox70.ss()(j, i);
    }
  }

  // MOX 8.7%
  //         Total       Transport   Absorption  Capture     Fission     Nu          Chi
  // Group
  // 1 2.16280E-01 1.83045E-01 9.48620E-03 8.14110E-04 8.67209E-03 2.90426E+00 5.87910E-01
  // Group
  // 2 3.61700E-01 3.36705E-01 4.65560E-03 3.03134E-03 1.62426E-03 2.91795E+00 4.11760E-01
  // Group
  // 3 5.05630E-01 5.00507E-01 3.62400E-02 2.59684E-02 1.02716E-02 2.86986E+00 3.39060E-04
  // Group
  // 4 6.11170E-01 6.06174E-01 1.32720E-01 9.36753E-02 3.90447E-02 2.87491E+00 1.17610E-07
  // Group 5 5.08900E-01 5.02754E-01 2.08400E-01 1.89142E-01 1.92576E-02 2.87175E+00
  // 0.00000E+00 Group
  // 6 9.26670E-01 9.21028E-01 6.58700E-01 2.83812E-01 3.74888E-01 2.86752E+00 0.00000E+00
  // Group 7 9.60990E-01 9.55231E-01 6.90170E-01 2.59571E-01 4.30599E-01 2.87808E+00
  // 0.00000E+00
  //
  // Scatter Matrix
  //       to Group 1  to Group 2  to Group 3  to Group 4  to Group 5  to Group 6  to
  //       Group 7
  // Group 1 1.31504E-01 4.20460E-02 8.69720E-06 5.19380E-09 0.00000E+00 0.00000E+00
  // 0.00000E+00 Group 2 0.00000E+00 3.30403E-01 1.64630E-03 2.60060E-09 0.00000E+00
  // 0.00000E+00 0.00000E+00 Group 3 0.00000E+00 0.00000E+00 4.61792E-01 2.47490E-03
  // 0.00000E+00 0.00000E+00 0.00000E+00 Group 4 0.00000E+00 0.00000E+00
  // 0.00000E+00 4.68021E-01 5.43300E-03 0.00000E+00 0.00000E+00 Group 5 0.00000E+00
  // 0.00000E+00 0.00000E+00 1.85970E-04 2.85771E-01 8.39730E-03 8.92800E-09 Group 6
  // 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 2.39160E-03 2.47614E-01 1.23220E-02
  // Group 7 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
  // 0.00000E+00 8.96810E-03 2.56093E-01

  XSec mox87(num_groups);
  mox87.isMacro() = true;
  mox87.isFissile() = true;
  mox87.a() = {9.48620e-03, 4.65560e-03, 3.62400e-02, 1.32720e-01,
               2.08400e-01, 6.58700e-01, 6.90170e-01};
  mox87.f() = {8.67209e-03, 1.62426e-03, 1.02716e-02, 3.90447e-02,
               1.92576e-02, 3.74888e-01, 4.30599e-01};
  nu = {2.90426, 2.91795, 2.86986, 2.87491, 2.87175, 2.86752, 2.87808};
  for (Int i = 0; i < num_groups; ++i) {
    mox87.nuf()[i] = nu[i] * mox87.f()[i];
  }
  mox87.tr() = {1.83045e-01, 3.36705e-01, 5.00507e-01, 6.06174e-01,
                5.02754e-01, 9.21028e-01, 9.55231e-01};
  // clang-format off
  mox87.ss().asVector() = {
    1.31504e-01, 4.20460e-02, 8.69720e-06, 5.19380e-09, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 3.30403e-01, 1.64630e-03, 2.60060e-09, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 4.61792e-01, 2.47490e-03, 0.00000e+00, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 4.68021e-01, 5.43300e-03, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 1.85970e-04, 2.85771e-01, 8.39730e-03, 8.92800e-09,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.39160e-03, 2.47614e-01, 1.23220e-02,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 8.96810e-03, 2.56093e-01};
  // clang-format on
  for (Int i = 0; i < num_groups; ++i) {
    for (Int j = 0; j < num_groups; ++j) {
      mox87.s()[i] += mox87.ss()(j, i);
    }
  }

  // Fission Chamber
  //         Total       Transport   Absorption  Capture     Fission     Nu          Chi
  // Group
  // 1 1.90730E-01 1.26032E-01 5.11320E-04 5.11315E-04 4.79002E-09 2.76283E+00 5.87910E-01
  // Group
  // 2 4.56520E-01 2.93160E-01 7.58130E-05 7.58072E-05 5.82564E-09 2.46239E+00 4.11760E-01
  // Group
  // 3 6.40700E-01 2.84250E-01 3.16430E-04 3.15966E-04 4.63719E-07 2.43380E+00 3.39060E-04
  // Group
  // 4 6.49840E-01 2.81020E-01 1.16750E-03 1.16226E-03 5.24406E-06 2.43380E+00 1.17610E-07
  // Group 5 6.70630E-01 3.34460E-01 3.39770E-03 3.39755E-03 1.45390E-07 2.43380E+00
  // 0.00000E+00 Group
  // 6 8.75060E-01 5.65640E-01 9.18860E-03 9.18789E-03 7.14972E-07 2.43380E+00 0.00000E+00
  // Group 7 1.43450E+00 1.17214E+00 2.32440E-02 2.32419E-02 2.08041E-06 2.43380E+00
  // 0.00000E+00
  //
  // Scatter Matrix
  //       to Group 1  to Group 2  to Group 3  to Group 4  to Group 5  to Group 6  to
  //       Group 7
  // Group 1 6.61659E-02 5.90700E-02 2.83340E-04 1.46220E-06 2.06420E-08 0.00000E+00
  // 0.00000E+00 Group 2
  // 0.00000E+00 2.40377E-01 5.24350E-02 2.49900E-04 1.92390E-05 2.98750E-06 4.21400E-07
  // Group 3 0.00000E+00
  // 0.00000E+00 1.83425E-01 9.22880E-02 6.93650E-03 1.07900E-03 2.05430E-04 Group 4
  // 0.00000E+00 0.00000E+00 0.00000E+00 7.90769E-02 1.69990E-01 2.58600E-02 4.92560E-03
  // Group 5 0.00000E+00 0.00000E+00
  // 0.00000E+00 3.73400E-05 9.97570E-02 2.06790E-01 2.44780E-02 Group 6 0.00000E+00
  // 0.00000E+00 0.00000E+00 0.00000E+00 9.17420E-04 3.16774E-01 2.38760E-01 Group 7
  // 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 4.97930E-02 1.09910E+00

  XSec fc(num_groups);
  fc.isMacro() = true;
  fc.isFissile() = true;
  fc.a() = {5.11320e-04, 7.58130e-05, 3.16430e-04, 1.16750e-03,
            3.39770e-03, 9.18860e-03, 2.32440e-02};
  fc.f() = {4.79002e-09, 5.82564e-09, 4.63719e-07, 5.24406e-06,
            1.45390e-07, 7.14972e-07, 2.08041e-06};
  nu = {2.76283, 2.46239, 2.43380, 2.43380, 2.43380, 2.43380, 2.43380};
  for (Int i = 0; i < num_groups; ++i) {
    fc.nuf()[i] = nu[i] * fc.f()[i];
  }
  fc.tr() = {1.26032e-01, 2.93160e-01, 2.84250e-01, 2.81020e-01,
             3.34460e-01, 5.65640e-01, 1.17214e+00};
  // clang-format off
  fc.ss().asVector() = {
    6.61659e-02, 5.90700e-02, 2.83340e-04, 1.46220e-06, 2.06420e-08, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 2.40377e-01, 5.24350e-02, 2.49900e-04, 1.92390e-05, 2.98750e-06, 4.21400e-07,
    0.00000e+00, 0.00000e+00, 1.83425e-01, 9.22880e-02, 6.93650e-03, 1.07900e-03, 2.05430e-04,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 7.90769e-02, 1.69990e-01, 2.58600e-02, 4.92560e-03,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 3.73400e-05, 9.97570e-02, 2.06790e-01, 2.44780e-02,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.17420e-04, 3.16774e-01, 2.38760e-01,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 4.97930e-02, 1.09910e+00};
  // clang-format on
  for (Int i = 0; i < num_groups; ++i) {
    for (Int j = 0; j < num_groups; ++j) {
      fc.s()[i] += fc.ss()(j, i);
    }
  }

  // Guide Tube
  //         Total       Transport   Absorption  Capture
  // Group 1 1.90730E-01 1.26032E-01 5.11320E-04 5.11320E-04
  // Group 2 4.56520E-01 2.93160E-01 7.58010E-05 7.58010E-05
  // Group 3 6.40670E-01 2.84240E-01 3.15720E-04 3.15720E-04
  // Group 4 6.49670E-01 2.80960E-01 1.15820E-03 1.15820E-03
  // Group 5 6.70580E-01 3.34440E-01 3.39750E-03 3.39750E-03
  // Group 6 8.75050E-01 5.65640E-01 9.18780E-03 9.18780E-03
  // Group 7 1.43450E+00 1.17215E+00 2.32420E-02 2.32420E-02
  //
  // Scatter Matrix
  //      to Group 1  to Group 2  to Group 3  to Group 4  to Group 5  to Group 6  to Group
  //      7
  // Group 1 6.61659E-02 5.90700E-02 2.83340E-04 1.46220E-06 2.06420E-08 0.00000E+00
  // 0.00000E+00 Group 2
  // 0.00000E+00 2.40377E-01 5.24350E-02 2.49900E-04 1.92390E-05 2.98750E-06 4.21400E-07
  // Group 3 0.00000E+00
  // 0.00000E+00 1.83297E-01 9.23970E-02 6.94460E-03 1.08030E-03 2.05670E-04 Group 4
  // 0.00000E+00 0.00000E+00 0.00000E+00 7.88511E-02 1.70140E-01 2.58810E-02 4.92970E-03
  // Group 5 0.00000E+00 0.00000E+00
  // 0.00000E+00 3.73330E-05 9.97372E-02 2.06790E-01 2.44780E-02 Group 6 0.00000E+00
  // 0.00000E+00 0.00000E+00 0.00000E+00 9.17260E-04 3.16765E-01 2.38770E-01 Group 7
  // 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 4.97920E-02 1.09912E+00

  XSec gt(num_groups);
  gt.isMacro() = true;
  gt.a() = {5.11320e-04, 7.58010e-05, 3.15720e-04, 1.15820e-03,
            3.39750e-03, 9.18780e-03, 2.32420e-02};
  gt.tr() = {1.26032e-01, 2.93160e-01, 2.84240e-01, 2.80960e-01,
             3.34440e-01, 5.65640e-01, 1.17215e+00};
  // clang-format off
  gt.ss().asVector() = {
    6.61659e-02, 5.90700e-02, 2.83340e-04, 1.46220e-06, 2.06420e-08, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 2.40377e-01, 5.24350e-02, 2.49900e-04, 1.92390e-05, 2.98750e-06, 4.21400e-07,
    0.00000e+00, 0.00000e+00, 1.83297e-01, 9.23970e-02, 6.94460e-03, 1.08030e-03, 2.05670e-04,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 7.88511e-02, 1.70140e-01, 2.58810e-02, 4.92970e-03,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 3.73330e-05, 9.97372e-02, 2.06790e-01, 2.44780e-02,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 9.17260e-04, 3.16765e-01, 2.38770e-01,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 4.97920e-02, 1.09912e+00};
  // clang-format on
  for (Int i = 0; i < num_groups; ++i) {
    for (Int j = 0; j < num_groups; ++j) {
      gt.s()[i] += gt.ss()(j, i);
    }
  }

  // Moderator
  //         Total       Transport   Absorption  Capture
  // Group 1 2.30070E-01 1.59206E-01 6.01050E-04 6.01050E-04
  // Group 2 7.76460E-01 4.12970E-01 1.57930E-05 1.57930E-05
  // Group 3 1.48420E+00 5.90310E-01 3.37160E-04 3.37160E-04
  // Group 4 1.50520E+00 5.84350E-01 1.94060E-03 1.94060E-03
  // Group 5 1.55920E+00 7.18000E-01 5.74160E-03 5.74160E-03
  // Group 6 2.02540E+00 1.25445E+00 1.50010E-02 1.50010E-02
  // Group 7 3.30570E+00 2.65038E+00 3.72390E-02 3.72390E-02
  //
  // Scatter Matrix
  //     to Group 1  to Group 2  to Group 3  to Group 4  to Group 5  to Group 6  to Group
  //     7
  // Group 1 4.44777E-02 1.13400E-01 7.23470E-04 3.74990E-06 5.31840E-08 0.00000E+00
  // 0.00000E+00 Group 2
  // 0.00000E+00 2.82334E-01 1.29940E-01 6.23400E-04 4.80020E-05 7.44860E-06 1.04550E-06
  // Group 3 0.00000E+00
  // 0.00000E+00 3.45256E-01 2.24570E-01 1.69990E-02 2.64430E-03 5.03440E-04 Group 4
  // 0.00000E+00 0.00000E+00 0.00000E+00 9.10284E-02 4.15510E-01 6.37320E-02 1.21390E-02
  // Group 5 0.00000E+00 0.00000E+00
  // 0.00000E+00 7.14370E-05 1.39138E-01 5.11820E-01 6.12290E-02 Group 6 0.00000E+00
  // 0.00000E+00 0.00000E+00 0.00000E+00 2.21570E-03 6.99913E-01 5.37320E-01 Group 7
  // 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 1.32440E-01 2.48070E+00

  XSec mo(num_groups);
  mo.isMacro() = true;
  mo.a() = {6.01050e-04, 1.57930e-05, 3.37160e-04, 1.94060e-03,
            5.74160e-03, 1.50010e-02, 3.72390e-02};
  mo.tr() = {1.59206e-01, 4.12970e-01, 5.90310e-01, 5.84350e-01,
             7.18000e-01, 1.25445e+00, 2.65038e+00};
  // clang-format off
  mo.ss().asVector() = {
    4.44777e-02, 1.13400e-01, 7.23470e-04, 3.74990e-06, 5.31840e-08, 0.00000e+00, 0.00000e+00,
    0.00000e+00, 2.82334e-01, 1.29940e-01, 6.23400e-04, 4.80020e-05, 7.44860e-06, 1.04550e-06,
    0.00000e+00, 0.00000e+00, 3.45256e-01, 2.24570e-01, 1.69990e-02, 2.64430e-03, 5.03440e-04,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 9.10284e-02, 4.15510e-01, 6.37320e-02, 1.21390e-02,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 7.14370e-05, 1.39138e-01, 5.11820e-01, 6.12290e-02,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 2.21570e-03, 6.99913e-01, 5.37320e-01,
    0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 1.32440e-01, 2.48070e+00};
  // clang-format on
  for (Int i = 0; i < num_groups; ++i) {
    for (Int j = 0; j < num_groups; ++j) {
      mo.s()[i] += mo.ss()(j, i);
    }
  }

  return {uo2, mox43, mox70, mox87, fc, gt, mo};
}

} // namespace um2
