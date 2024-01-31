#include <um2/math/angular_quadrature.hpp>

#include "../test_macros.hpp"

F constexpr eps = condCast<F>(1e-6);

HOSTDEV
TEST_CASE(chebyshev_chebyshev)
{
  F constexpr half = static_cast<F>(1) / static_cast<F>(2);
  auto const type = um2::AngularQuadratureType::Chebyshev;
  um2::ProductAngularQuadrature const q(type, 1, type, 2);
  ASSERT(q.azimuthalDegree() == 1);
  ASSERT(q.polarDegree() == 2);
  ASSERT_NEAR(q.azimuthalWeights()[0], static_cast<F>(1), eps);
  ASSERT_NEAR(q.azimuthalAngles()[0], um2::pi_4<F>, eps);
  ASSERT_NEAR(q.polarWeights()[0], half, eps);
  ASSERT_NEAR(q.polarWeights()[1], half, eps);
  ASSERT_NEAR(q.polarAngles()[0], um2::pi<F> / static_cast<F>(8), eps);
  ASSERT_NEAR(q.polarAngles()[1], um2::pi<F> * static_cast<F>(3) / static_cast<F>(8),
              eps);
}

//=============================================================================
// CUDA
//=============================================================================

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(chebyshev_chebyshev);
#endif

TEST_SUITE(AngularQuadrature) { TEST_HOSTDEV(chebyshev_chebyshev); }

auto
main() -> int
{
  RUN_SUITE(AngularQuadrature);
  return 0;
}
