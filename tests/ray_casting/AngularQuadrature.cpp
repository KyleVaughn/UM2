#include <um2/ray_casting/AngularQuadrature.hpp>

#include "../test_macros.hpp"

template <typename T>
HOSTDEV
TEST_CASE(chebyshev_chebyshev)
{
  auto const type = um2::AngularQuadratureType::Chebyshev;
  um2::ProductAngularQuadrature<T> const q(type, 1, type, 2);
  assert(q.wazi.size() == 1);
  assert(q.azi.size() == 1);
  assert(q.wpol.size() == 2);
  assert(q.pol.size() == 2);
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
