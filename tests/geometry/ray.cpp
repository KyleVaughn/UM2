#include <um2/geometry/ray.hpp>

#include "../test_macros.hpp"

//=============================================================================
// Constructors
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(constructor_origin_direction)
{
  um2::Point<D> origin = um2::Point<D>::zero();
  um2::Vec<D, Float> direction = um2::Vec<D, Float>::zero();
  origin[0] = static_cast<Float>(1);
  direction[1] = static_cast<Float>(1);

  um2::Ray<D> const ray(origin, direction);
  ASSERT(um2::isApprox(ray.origin(), origin));
  ASSERT(um2::isApprox(ray.direction(), direction));
}

//=============================================================================
// Interpolation
//=============================================================================

template <Int D>
HOSTDEV
TEST_CASE(interpolate)
{
  um2::Point<D> origin = um2::Point<D>::zero();
  um2::Vec<D, Float> direction = um2::Vec<D, Float>::zero();
  origin[0] = static_cast<Float>(1);
  direction[1] = static_cast<Float>(1);

  um2::Ray<D> const ray(origin, direction);
  ASSERT(um2::isApprox(ray(static_cast<Float>(0)), origin));
  ASSERT(um2::isApprox(ray(static_cast<Float>(1)), origin + direction));
}

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(constructor_origin_direction, D);

template <Int D>
MAKE_CUDA_KERNEL(interpolate, D);
#endif

template <Int D>
TEST_SUITE(Ray)
{
  TEST_HOSTDEV(constructor_origin_direction, D);
  TEST_HOSTDEV(interpolate, D);
}

auto
main() -> int
{
  RUN_SUITE(Ray<2>);
  RUN_SUITE(Ray<3>);
  return 0;
}
