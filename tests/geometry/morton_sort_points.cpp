#include <um2/config.hpp>
#include <um2/geometry/morton_sort_points.hpp>
#include <um2/geometry/point.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

template <class T>
TEST_CASE(mortonSort2D)
{
  // Map a 4 by 4 grid of points to the unit square and sort them by Morton code.
  um2::Vector<um2::Point2<T>> points(16);
  for (Int i = 0; i < 4; ++i) {
    for (Int j = 0; j < 4; ++j) {
      points[i * 4 + j] = um2::Point2<T>(i, j);
      points[i * 4 + j] /= 3;
    }
  }
  um2::mortonSort<2>(points.begin(), points.end());
  ASSERT(points[0].isApprox(um2::Point2<T>(0, 0) / 3));
  ASSERT(points[1].isApprox(um2::Point2<T>(1, 0) / 3));
  ASSERT(points[2].isApprox(um2::Point2<T>(0, 1) / 3));
  ASSERT(points[3].isApprox(um2::Point2<T>(1, 1) / 3));
  ASSERT(points[4].isApprox(um2::Point2<T>(2, 0) / 3));
  ASSERT(points[5].isApprox(um2::Point2<T>(3, 0) / 3));
  ASSERT(points[6].isApprox(um2::Point2<T>(2, 1) / 3));
  ASSERT(points[7].isApprox(um2::Point2<T>(3, 1) / 3));
  ASSERT(points[8].isApprox(um2::Point2<T>(0, 2) / 3));
  ASSERT(points[9].isApprox(um2::Point2<T>(1, 2) / 3));
  ASSERT(points[10].isApprox(um2::Point2<T>(0, 3) / 3));
  ASSERT(points[11].isApprox(um2::Point2<T>(1, 3) / 3));
  ASSERT(points[12].isApprox(um2::Point2<T>(2, 2) / 3));
  ASSERT(points[13].isApprox(um2::Point2<T>(3, 2) / 3));
  ASSERT(points[14].isApprox(um2::Point2<T>(2, 3) / 3));
  ASSERT(points[15].isApprox(um2::Point2<T>(3, 3) / 3));
}

template <class T>
TEST_CASE(mortonSort3D)
{
  // Map a 4 by 4 by 4 grid of points to the unit cube and sort them by Morton code.
  um2::Vector<um2::Point3<T>> points(64);
  for (Int i = 0; i < 4; ++i) {
    for (Int j = 0; j < 4; ++j) {
      for (Int k = 0; k < 4; ++k) {
        points[i * 16 + j * 4 + k] = um2::Point3<T>(i, j, k);
        points[i * 16 + j * 4 + k] /= 3;
      }
    }
  }
  um2::mortonSort<3>(points.begin(), points.end());
  ASSERT(points[0].isApprox(um2::Point3<T>(0, 0, 0) / 3));
  ASSERT(points[1].isApprox(um2::Point3<T>(1, 0, 0) / 3));
  ASSERT(points[2].isApprox(um2::Point3<T>(0, 1, 0) / 3));
  ASSERT(points[3].isApprox(um2::Point3<T>(1, 1, 0) / 3));
  ASSERT(points[4].isApprox(um2::Point3<T>(0, 0, 1) / 3));
  ASSERT(points[5].isApprox(um2::Point3<T>(1, 0, 1) / 3));
  ASSERT(points[6].isApprox(um2::Point3<T>(0, 1, 1) / 3));
  ASSERT(points[7].isApprox(um2::Point3<T>(1, 1, 1) / 3));
  ASSERT(points[8].isApprox(um2::Point3<T>(2, 0, 0) / 3));
  ASSERT(points[9].isApprox(um2::Point3<T>(3, 0, 0) / 3));
  ASSERT(points[10].isApprox(um2::Point3<T>(2, 1, 0) / 3));
  ASSERT(points[11].isApprox(um2::Point3<T>(3, 1, 0) / 3));
  ASSERT(points[12].isApprox(um2::Point3<T>(2, 0, 1) / 3));
  ASSERT(points[13].isApprox(um2::Point3<T>(3, 0, 1) / 3));
  ASSERT(points[14].isApprox(um2::Point3<T>(2, 1, 1) / 3));
  ASSERT(points[15].isApprox(um2::Point3<T>(3, 1, 1) / 3));
}

template <class T>
TEST_SUITE(mortonSort)
{
  TEST(mortonSort2D<T>);
  TEST(mortonSort3D<T>);
}

auto
main() -> int
{
  RUN_SUITE(mortonSort<float>);
  RUN_SUITE(mortonSort<double>);
  return 0;
}
