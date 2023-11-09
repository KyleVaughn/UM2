#include <um2/math/angular_quadrature.hpp>

#include "../test_macros.hpp"

template <typename T>
HOSTDEV
TEST_CASE(chebyshev_chebyshev)
{
  auto const type = um2::AngularQuadratureType::Chebyshev;
  um2::ProductAngularQuadrature<T> const q(type, 1, type, 2);
  ASSERT(q.wazi.size() == 1);
  ASSERT(q.azi.size() == 1);
  ASSERT(q.wpol.size() == 2);
  ASSERT(q.pol.size() == 2);
  ASSERT_NEAR(q.wazi[0], static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(q.azi[0], um2::pi<T> / static_cast<T>(4), static_cast<T>(1e-6));
  ASSERT_NEAR(q.wpol[0], static_cast<T>(0.5), static_cast<T>(1e-6));
  ASSERT_NEAR(q.wpol[1], static_cast<T>(0.5), static_cast<T>(1e-6));
  ASSERT_NEAR(q.pol[0], um2::pi<T> / static_cast<T>(8), static_cast<T>(1e-6));
  ASSERT_NEAR(q.pol[1], um2::pi<T> * static_cast<T>(3) / static_cast<T>(8),
              static_cast<T>(1e-6));
}

//=============================================================================
// CUDA
//=============================================================================

#if UM2_USE_CUDA
template <typename T>
MAKE_CUDA_KERNEL(chebyshev_chebyshev, T);
#endif

template <typename T>
TEST_SUITE(AngularQuadrature)
{
  TEST_HOSTDEV(chebyshev_chebyshev, 1, 1, T);
}

auto
main() -> int
{
  RUN_SUITE((AngularQuadrature<float>));
  RUN_SUITE((AngularQuadrature<double>));
  return 0;
}
