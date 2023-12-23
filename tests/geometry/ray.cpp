#include <um2/geometry/ray.hpp>

#include "../test_macros.hpp"

//=============================================================================
// Constructors
//=============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(constructor_origin_direction)
{
  um2::Point<D, T> origin = um2::Point<D, T>::zero();
  um2::Vec<D, T> direction = um2::Vec<D, T>::zero();
  origin[0] = static_cast<T>(1);
  direction[1] = static_cast<T>(1);

  um2::Ray<D, T> const ray(origin, direction);
  ASSERT(um2::isApprox(ray.origin(), origin));
  ASSERT(um2::isApprox(ray.direction(), direction));
}

//=============================================================================
// Interpolation
//=============================================================================

template <Size D, typename T>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Point<D, T> origin = um2::Point<D, T>::zero();
  um2::Vec<D, T> direction = um2::Vec<D, T>::zero();
  origin[0] = static_cast<T>(1);
  direction[1] = static_cast<T>(1);

  um2::Ray<D, T> const ray(origin, direction);
  ASSERT(um2::isApprox(ray(static_cast<T>(0)), origin));
  ASSERT(um2::isApprox(ray(static_cast<T>(1)), origin + direction));
}

#if UM2_USE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(constructor_origin_direction, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(interpolate, D, T);
#endif

template <Size D, typename T>
TEST_SUITE(Ray)
{
  TEST_HOSTDEV(constructor_origin_direction, 1, 1, D, T);
  TEST_HOSTDEV(interpolate, 1, 1, D, T);
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
