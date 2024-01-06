#include <um2/geometry/modular_rays.hpp>

#include <um2/common/log.hpp>

#include "../test_macros.hpp"

// Test problem: For a 2 by 2 grid, for a few test angles, ensure that a sample
// ray is both cyclic and modular.
//                   r_stop
// 4 +--------------x-+----------------+
//   |             /  |                |
//   |            /   |                |
//   |           /    |                |
//   |          /     |                |
// 2 +---------/------+----------------+
//   |        /       |                |
//   |       /        |                |
//   |      /         |                |
//   |     /          |                |
// 0 +----o-----------+----------------+
//   0    r_start     3                6
//

template <typename T>
HOSTDEV
TEST_CASE(modular_ray_params)
{
  um2::Point2<T> const p00(0, 0);
  um2::Point2<T> const p30(3, 0);
  um2::Point2<T> const p02(0, 2);
  um2::Point2<T> const p32(3, 2);
  um2::Point2<T> const p62(6, 2);
  um2::Point2<T> const p34(3, 4);
  um2::Point2<T> const p64(6, 4);
  um2::AxisAlignedBox2<T> const box00(p00, p32);
  um2::AxisAlignedBox2<T> const box10(p30, p62);
  um2::AxisAlignedBox2<T> const box01(p02, p34);
  um2::AxisAlignedBox2<T> const box11(p32, p64);
  um2::Vector<um2::AxisAlignedBox2<T>> const boxes{box00, box10, box01, box11};

  Size const degree = 2; // azimuthal angles per quadrant
  T const pi_deg = um2::pi_4<T> / degree;
  // pi/8, 3pi/8, 5pi/8, 7pi/8
  um2::Vector<T> const angles{pi_deg, 3 * pi_deg, 5 * pi_deg, 7 * pi_deg};

  T const s = static_cast<T>(0.5); // ray spacing
  um2::Vector<um2::ModularRayParams<T>> ray_params(4);
  for (Size i = 0; i < 4; ++i) {
    ray_params[i] = um2::ModularRayParams(angles[0], s, boxes[i]);
  }

  // Ensure that all the ray params have the same num_rays, spacing, and direction.
  Size const nx = ray_params[0].getNumXRays();
  Size const ny = ray_params[0].getNumYRays();
  for (Size i = 1; i < 4; ++i) {
    ASSERT(ray_params[i].getNumXRays() == nx);
    ASSERT(ray_params[i].getNumYRays() == ny);
    ASSERT(um2::isApprox(ray_params[i].getSpacing(), ray_params[0].getSpacing()));
    ASSERT(um2::isApprox(ray_params[i].getDirection(), ray_params[0].getDirection()));
  }

  // The first ray on the x-axis in box00 should become the first ray on the
  // y-axis of box10
  auto const r00 = ray_params[0].getRay(0);
  auto const intersection0 = um2::intersect(r00, box00);
  auto const p1 = r00(intersection0[1]);
  auto const r10 = ray_params[1].getRay(nx);
  ASSERT(um2::isApprox(p1, r10.origin()));

  // Check that the complementary angle shares the same ray origin.
  um2::ModularRayParams const params(angles[3], s, boxes[0]);
  auto const ra0 = params.getRay(params.getNumXRays() - 1);
  ASSERT(um2::isApprox(ra0.origin(), r00.origin()));
}

#if UM2_USE_CUDA
template <typename T>
MAKE_CUDA_KERNEL(modular_ray_params, T);
#endif

template <typename T>
TEST_SUITE(ModularRayParams)
{
  TEST_HOSTDEV(modular_ray_params, 1, 1, T);
}

auto
main() -> int
{
  RUN_SUITE((ModularRayParams<float>));
  RUN_SUITE((ModularRayParams<double>));
  return 0;
}
