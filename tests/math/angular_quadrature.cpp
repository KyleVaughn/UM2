#include <um2/math/angular_quadrature.hpp>

#include "../test_macros.hpp"

template <typename T>
HOSTDEV
TEST_CASE(chebyshev_chebyshev)
{
  T constexpr eps = static_cast<T>(1e-6);

  auto const type = um2::AngularQuadratureType::Chebyshev;
  um2::ProductAngularQuadrature<T> const q(type, 1, type, 2);
  ASSERT(q.azimuthalDegree() == 1);
  ASSERT(q.polarDegree() == 2);
  ASSERT_NEAR(q.azimuthalWeights()[0], static_cast<T>(1), eps);
  ASSERT_NEAR(q.azimuthalAngles()[0], um2::pi_4<T>, eps);
  ASSERT_NEAR(q.polarWeights()[0], static_cast<T>(0.5), eps);
  ASSERT_NEAR(q.polarWeights()[1], static_cast<T>(0.5), eps);
  ASSERT_NEAR(q.polarAngles()[0], um2::pi<T> / static_cast<T>(8), eps);
  ASSERT_NEAR(q.polarAngles()[1], um2::pi<T> * static_cast<T>(3) / static_cast<T>(8), eps);
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
