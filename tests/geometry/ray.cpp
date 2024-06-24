#include <um2/config.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/ray.hpp>

#include "../test_macros.hpp"

//=============================================================================
// Constructors
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(constructor_origin_direction)
{
  um2::Point<D, T> origin = um2::Point<D, T>::zero();
  um2::Point<D, T> direction = um2::Point<D, T>::zero();
  origin[0] = static_cast<T>(1);
  direction[1] = static_cast<T>(1);

  um2::Ray<D, T> const ray(origin, direction);
  ASSERT(ray.origin().isApprox(origin));
  ASSERT(ray.direction().isApprox(direction));
}

//=============================================================================
// Interpolation
//=============================================================================

template <Int D, class T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Point<D, T> origin = um2::Point<D, T>::zero();
  um2::Point<D, T> direction = um2::Point<D, T>::zero();
  origin[0] = static_cast<T>(1);
  direction[1] = static_cast<T>(1);

  um2::Ray<D, T> const ray(origin, direction);
  ASSERT(ray(static_cast<T>(0)).isApprox(origin));
  ASSERT(ray(static_cast<T>(1)).isApprox(origin + direction));
}

#if UM2_USE_CUDA
template <Int D, class T>
MAKE_CUDA_KERNEL(constructor_origin_direction, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(interpolate, D, T);
#endif

template <Int D, class T>
TEST_SUITE(Ray)
{
  TEST_HOSTDEV(constructor_origin_direction, D, T);
  TEST_HOSTDEV(interpolate, D, T);
}

auto
main() -> int
{
  RUN_SUITE((Ray<2, float>));
  RUN_SUITE((Ray<3, float>));

  RUN_SUITE((Ray<2, double>));
  RUN_SUITE((Ray<3, double>));
  return 0;
}
