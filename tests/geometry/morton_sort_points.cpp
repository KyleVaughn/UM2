#include <um2/geometry/morton_sort_points.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

TEST_CASE(mortonSort2D)
{
  // Map a 4 by 4 grid of points to the unit square and sort them by Morton code.
  um2::Vector<um2::Point2> points(16);
  for (Int i = 0; i < 4; ++i) {
    for (Int j = 0; j < 4; ++j) {
      points[i * 4 + j] = um2::Point2(i, j);
      points[i * 4 + j] /= 3;
    }
  }
  um2::mortonSort<2>(points.begin(), points.end());
  ASSERT(um2::isApprox<2>(points[0], um2::Point2(0, 0) / 3));
  ASSERT(um2::isApprox<2>(points[1], um2::Point2(1, 0) / 3));
  ASSERT(um2::isApprox<2>(points[2], um2::Point2(0, 1) / 3));
  ASSERT(um2::isApprox<2>(points[3], um2::Point2(1, 1) / 3));
  ASSERT(um2::isApprox<2>(points[4], um2::Point2(2, 0) / 3));
  ASSERT(um2::isApprox<2>(points[5], um2::Point2(3, 0) / 3));
  ASSERT(um2::isApprox<2>(points[6], um2::Point2(2, 1) / 3));
  ASSERT(um2::isApprox<2>(points[7], um2::Point2(3, 1) / 3));
  ASSERT(um2::isApprox<2>(points[8], um2::Point2(0, 2) / 3));
  ASSERT(um2::isApprox<2>(points[9], um2::Point2(1, 2) / 3));
  ASSERT(um2::isApprox<2>(points[10], um2::Point2(0, 3) / 3));
  ASSERT(um2::isApprox<2>(points[11], um2::Point2(1, 3) / 3));
  ASSERT(um2::isApprox<2>(points[12], um2::Point2(2, 2) / 3));
  ASSERT(um2::isApprox<2>(points[13], um2::Point2(3, 2) / 3));
  ASSERT(um2::isApprox<2>(points[14], um2::Point2(2, 3) / 3));
  ASSERT(um2::isApprox<2>(points[15], um2::Point2(3, 3) / 3));
}

TEST_CASE(mortonSort3D)
{
  // Map a 4 by 4 by 4 grid of points to the unit cube and sort them by Morton code.
  um2::Vector<um2::Point3> points(64);
  for (Int i = 0; i < 4; ++i) {
    for (Int j = 0; j < 4; ++j) {
      for (Int k = 0; k < 4; ++k) {
        points[i * 16 + j * 4 + k] = um2::Point3(i, j, k);
        points[i * 16 + j * 4 + k] /= 3;
      }
    }
  }
  um2::mortonSort<3>(points.begin(), points.end());
  ASSERT(um2::isApprox<3>(points[0], um2::Point3(0, 0, 0) / 3));
  ASSERT(um2::isApprox<3>(points[1], um2::Point3(1, 0, 0) / 3));
  ASSERT(um2::isApprox<3>(points[2], um2::Point3(0, 1, 0) / 3));
  ASSERT(um2::isApprox<3>(points[3], um2::Point3(1, 1, 0) / 3));
  ASSERT(um2::isApprox<3>(points[4], um2::Point3(0, 0, 1) / 3));
  ASSERT(um2::isApprox<3>(points[5], um2::Point3(1, 0, 1) / 3));
  ASSERT(um2::isApprox<3>(points[6], um2::Point3(0, 1, 1) / 3));
  ASSERT(um2::isApprox<3>(points[7], um2::Point3(1, 1, 1) / 3));
  ASSERT(um2::isApprox<3>(points[8], um2::Point3(2, 0, 0) / 3));
  ASSERT(um2::isApprox<3>(points[9], um2::Point3(3, 0, 0) / 3));
  ASSERT(um2::isApprox<3>(points[10], um2::Point3(2, 1, 0) / 3));
  ASSERT(um2::isApprox<3>(points[11], um2::Point3(3, 1, 0) / 3));
  ASSERT(um2::isApprox<3>(points[12], um2::Point3(2, 0, 1) / 3));
  ASSERT(um2::isApprox<3>(points[13], um2::Point3(3, 0, 1) / 3));
  ASSERT(um2::isApprox<3>(points[14], um2::Point3(2, 1, 1) / 3));
  ASSERT(um2::isApprox<3>(points[15], um2::Point3(3, 1, 1) / 3));
}

TEST_SUITE(mortonSort)
{
  TEST(mortonSort2D);
  TEST(mortonSort3D);
}

auto
main() -> int
{
  RUN_SUITE(mortonSort);
  return 0;
}
