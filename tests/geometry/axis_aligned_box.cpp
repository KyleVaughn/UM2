#include <um2/config.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/stdlib/vector.hpp>

#include "../test_macros.hpp"

#include <cstdlib>

template <class T>
inline constexpr T eps = um2::epsDistance<T>();

template <class T>
inline constexpr T half = static_cast<T>(1) / static_cast<T>(2);

// Compiler complains when making this a static_assert, so we silence the warning
// NOLINTBEGIN(cert-dcl03-c,misc-static-assert)

template <Int D, class T>
HOSTDEV constexpr auto
makeBox() -> um2::AxisAlignedBox<D, T>
{
  um2::Point<D, T> minima;
  um2::Point<D, T> maxima;
  for (Int i = 0; i < D; ++i) {
    minima[i] = static_cast<T>(i);
    maxima[i] = static_cast<T>(i + 1);
  }
  return {minima, maxima};
}

template <Int D, class T>
HOSTDEV
TEST_CASE(extents)
{
  um2::AxisAlignedBox<D, T> const box = makeBox<D, T>();
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(box.extents(i), 1, eps<T>);
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::AxisAlignedBox<D, T> const box = makeBox<D, T>();
  um2::Point<D, T> p = box.centroid();
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(p[i], static_cast<T>(i) + half<T>, eps<T>);
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(contains)
{
  um2::AxisAlignedBox<D, T> const box = makeBox<D, T>();
  um2::Point<D, T> p;
  for (Int i = 0; i < D; ++i) {
    p[i] = static_cast<T>(i);
  }
  ASSERT(box.contains(p));
  p[0] += half<T> * eps<T>;
  ASSERT(box.contains(p));
  p[0] -= static_cast<T>(5) * eps<T>;
  ASSERT(!box.contains(p));
}

template <Int D, class T>
HOSTDEV
TEST_CASE(is_approx)
{
  um2::AxisAlignedBox<D, T> const box1 = makeBox<D, T>();
  um2::AxisAlignedBox<D, T> box2 = makeBox<D, T>();
  ASSERT(box1.isApprox(box2));
  auto p = box2.maxima();
  p += eps<T> / 2;
  box2 += p;
  ASSERT(box1.isApprox(box2));
  p += eps<T> * 10;
  box2 += p;
  ASSERT(!box1.isApprox(box2));
}

template <Int D, class T>
HOSTDEV
TEST_CASE(operator_plus)
{
  // (AxisAlignedBox, AxisAlignedBox)
  um2::AxisAlignedBox<D, T> const box = makeBox<D, T>();
  um2::Point<D, T> p0 = box.minima();
  um2::Point<D, T> p1 = box.maxima();
  for (Int i = 0; i < D; ++i) {
    p0[i] += static_cast<T>(1);
    p1[i] += static_cast<T>(1);
  }
  um2::AxisAlignedBox<D, T> const box2(p0, p1);
  um2::AxisAlignedBox<D, T> const box3 = box + box2;
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(box3.minima()[i], static_cast<T>(i), eps<T>);
    ASSERT_NEAR(box3.maxima()[i], static_cast<T>(i + 2), eps<T>);
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(bounding_box)
{
  // boundingBox(Point points[N])
  um2::Point<D, T> points[2 * D];
  for (Int i = 0; i < D; ++i) {
    um2::Point<D, T> p_right;
    um2::Point<D, T> p_left;
    for (Int j = 0; j < D; ++j) {
      p_right[j] = static_cast<T>(i + 1);
      p_left[j] = -static_cast<T>(i + 1);
    }
    points[static_cast<size_t>(2 * i + 0)] = p_right;
    points[static_cast<size_t>(2 * i + 1)] = p_left;
  }
  um2::AxisAlignedBox<D, T> const box4 = um2::boundingBox<D, T>(points, 2 * D);
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(box4.minima()[i], -static_cast<T>(D), eps<T>);
    ASSERT_NEAR(box4.maxima()[i], static_cast<T>(D), eps<T>);
  }
}

template <class T>
TEST_CASE(bounding_box_vector)
{
  // boundingBox(Vector<Point> points)
  Int const n = 20;
  um2::Vector<um2::Point2<T>> points(n);
  for (Int i = 0; i < n; ++i) {
    points[i][0] = static_cast<T>(i) / 10;
    points[i][1] = static_cast<T>(i) / 5;
  }
  auto const box = um2::boundingBox<2>(points.begin(), points.end());
  ASSERT_NEAR(box.minima()[0], static_cast<T>(0), eps<T>);
  ASSERT_NEAR(box.maxima()[0], static_cast<T>(n - 1) / 10, eps<T>);
  ASSERT_NEAR(box.minima()[1], static_cast<T>(0), eps<T>);
  ASSERT_NEAR(box.maxima()[1], static_cast<T>(n - 1) / 5, eps<T>);
}

template <class T>
HOSTDEV
TEST_CASE(intersect_ray)
{
  um2::AxisAlignedBox<2, T> const box = makeBox<2, T>();
  // up
  auto const ray0 =
      um2::Ray2<T>(um2::Point2<T>(half<T>, static_cast<T>(0)), um2::Point2<T>(0, 1));
  auto const intersection0 = box.intersect(ray0);
  ASSERT_NEAR(intersection0[0], static_cast<T>(1), eps<T>);
  ASSERT_NEAR(intersection0[1], static_cast<T>(2), eps<T>);

  auto const ray0_miss = um2::Ray2<T>(um2::Point2<T>(20, 0), um2::Point2<T>(0, -1));
  auto const intersection0_miss = box.intersect(ray0_miss);
  ASSERT(intersection0_miss[0] < 0);
  ASSERT(intersection0_miss[1] < 0);

  // right
  auto const ray1 =
      um2::Ray2<T>(um2::Point2<T>(-half<T>, half<T> * 3), um2::Point2<T>(1, 0));
  auto const intersection1 = box.intersect(ray1);
  ASSERT_NEAR(intersection1[0], half<T>, eps<T>);
  ASSERT_NEAR(intersection1[1], half<T> * 3, eps<T>);

  auto const ray1_miss = um2::Ray2<T>(um2::Point2<T>(0, 20), um2::Point2<T>(-1, 0));
  auto const intersection1_miss = box.intersect(ray1_miss);
  ASSERT(intersection1_miss[0] < 0);
  ASSERT(intersection1_miss[1] < 0);

  // 45 degrees
  auto const ray2 = um2::Ray2<T>(um2::Point2<T>(0, 1), um2::Point2<T>(1, 1).normalized());
  auto const intersection2 = box.intersect(ray2);
  ASSERT_NEAR(intersection2[0], static_cast<T>(0), eps<T>);
  ASSERT_NEAR(intersection2[1], um2::sqrt(static_cast<T>(2)), eps<T>);

  auto const ray2_miss =
      um2::Ray2<T>(um2::Point2<T>(0, 20), um2::Point2<T>(-1, -1).normalized());
  auto const intersection2_miss = box.intersect(ray2_miss);
  ASSERT(intersection2_miss[0] < 0);
  ASSERT(intersection2_miss[1] < 0);

  auto const ray3 = um2::Ray2<T>(um2::Point2<T>(half<T>, half<T> * 3),
                                 um2::Point2<T>(1, 1).normalized());
  auto const intersection3 = box.intersect(ray3);
  ASSERT_NEAR(intersection3[0], static_cast<T>(0), eps<T>);
  ASSERT_NEAR(intersection3[1], um2::sqrt(static_cast<T>(2)) / 2, eps<T>);

  auto const inv_dir = ray3.inverseDirection();
  auto const intersection4 = box.intersect(ray3, inv_dir);
  ASSERT_NEAR(intersection4[0], static_cast<T>(0), eps<T>);
  ASSERT_NEAR(intersection4[1], um2::sqrt(static_cast<T>(2)) / 2, eps<T>);
}

template <class T>
HOSTDEV
TEST_CASE(scale)
{
  um2::AxisAlignedBox<2, T> box(um2::Point2<T>(0, 0), um2::Point2<T>(1, 1));
  box.scale(2);
  ASSERT_NEAR(box.minima(0), castIfNot<T>(-0.5), eps<T>);
  ASSERT_NEAR(box.minima(1), castIfNot<T>(-0.5), eps<T>);
  ASSERT_NEAR(box.maxima(0), castIfNot<T>(1.5), eps<T>);
  ASSERT_NEAR(box.maxima(1), castIfNot<T>(1.5), eps<T>);
  ASSERT_NEAR(box.extents(0) * box.extents(1), castIfNot<T>(4), eps<T>);
  box.scale(castIfNot<T>(0.5));
  ASSERT_NEAR(box.minima(0), castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.minima(1), castIfNot<T>(0), eps<T>);
  ASSERT_NEAR(box.maxima(0), castIfNot<T>(1), eps<T>);
  ASSERT_NEAR(box.maxima(1), castIfNot<T>(1), eps<T>);
  ASSERT_NEAR(box.extents(0) * box.extents(1), castIfNot<T>(1), eps<T>);
}

template <class T>
HOSTDEV
TEST_CASE(intersects_box)
{
  // Simple intersection
  um2::AxisAlignedBox<2, T> a(um2::Point2<T>(0, 0), um2::Point2<T>(2, 2));
  um2::AxisAlignedBox<2, T> const b(um2::Point2<T>(1, 1), um2::Point2<T>(3, 3));
  ASSERT(a.intersects(b));
  ASSERT(b.intersects(a));
  a += um2::Point2<T>(-1, -1);
  a += um2::Point2<T>(4, 4);
  // a encompasses b
  ASSERT(a.intersects(b));
  ASSERT(b.intersects(a));
  um2::AxisAlignedBox<2, T> const c(um2::Point2<T>(1, 4), um2::Point2<T>(3, 5));
  // c is directly above b
  ASSERT(!b.intersects(c));
  ASSERT(!c.intersects(b));
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)

#if UM2_USE_CUDA
template <Int D, class T>
MAKE_CUDA_KERNEL(accessors, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(extents, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(centroid, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(contains, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(is_approx, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(operator_plus, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(bounding_box, D, T);

template <class T>
MAKE_CUDA_KERNEL(intersect_ray, T);
#endif

template <Int D, class T>
TEST_SUITE(aabb)
{
  TEST_HOSTDEV(extents, D, T);
  TEST_HOSTDEV(centroid, D, T);
  TEST_HOSTDEV(contains, D, T);
  TEST_HOSTDEV(is_approx, D, T);
  TEST_HOSTDEV(operator_plus, D, T);
  TEST_HOSTDEV(bounding_box, D, T);
  if constexpr (D == 2) {
    TEST(bounding_box_vector<T>);
    TEST_HOSTDEV(intersect_ray, T);
    TEST_HOSTDEV(scale, T);
    TEST_HOSTDEV(intersects_box, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((aabb<1, float>));
  RUN_SUITE((aabb<2, float>));
  RUN_SUITE((aabb<3, float>));

  RUN_SUITE((aabb<1, double>));
  RUN_SUITE((aabb<2, double>));
  RUN_SUITE((aabb<3, double>));
  return 0;
}
