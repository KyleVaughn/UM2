#include <um2/geometry/axis_aligned_box.hpp>

#include "../test_macros.hpp"

Float constexpr eps = um2::eps_distance;
Float constexpr half = static_cast<Float>(1) / static_cast<Float>(2);

// Compiler complains when making this a static_assert, so we silence the warning
// NOLINTBEGIN(cert-dcl03-c,misc-static-assert)

template <Int D>
HOSTDEV constexpr auto
makeBox() -> um2::AxisAlignedBox<D>
{
  um2::Point<D> minima;
  um2::Point<D> maxima;
  for (Int i = 0; i < D; ++i) {
    minima[i] = static_cast<Float>(i);
    maxima[i] = static_cast<Float>(i + 1);
  }
  return {minima, maxima};
}

template <Int D>
HOSTDEV
TEST_CASE(extents)
{
  um2::AxisAlignedBox<D> const box = makeBox<D>();
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(box.extents(i), 1, eps);
  }
}

template <Int D>
HOSTDEV
TEST_CASE(centroid)
{
  um2::AxisAlignedBox<D> const box = makeBox<D>();
  um2::Point<D> p = box.centroid();
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(p[i], static_cast<Float>(i) + half, eps);
  }
}

template <Int D>
HOSTDEV
TEST_CASE(contains)
{
  um2::AxisAlignedBox<D> const box = makeBox<D>();
  um2::Point<D> p;
  for (Int i = 0; i < D; ++i) {
    p[i] = static_cast<Float>(i);
  }
  ASSERT(box.contains(p));
  p[0] += half * eps;
  ASSERT(box.contains(p));
  p[0] -= static_cast<Float>(5) * eps;
  ASSERT(!box.contains(p));
}

template <Int D>
HOSTDEV
TEST_CASE(is_approx)
{
  um2::AxisAlignedBox<D> const box1 = makeBox<D>();
  um2::AxisAlignedBox<D> box2 = makeBox<D>();
  ASSERT(box1.isApprox(box2));
  um2::Point<D> p = box2.minima();
  for (Int i = 0; i < D; ++i) {
    p[i] = p[i] - eps / static_cast<Float>(10);
  }
  box2 += p;
  ASSERT(box1.isApprox(box2));
  box2 += 10 * p;
  ASSERT(!box1.isApprox(box2));
}

template <Int D>
HOSTDEV
TEST_CASE(operator_plus)
{
  // (AxisAlignedBox, AxisAlignedBox)
  um2::AxisAlignedBox<D> const box = makeBox<D>();
  um2::Point<D> p0 = box.minima();
  um2::Point<D> p1 = box.maxima();
  for (Int i = 0; i < D; ++i) {
    p0[i] += static_cast<Float>(1);
    p1[i] += static_cast<Float>(1);
  }
  um2::AxisAlignedBox<D> const box2(p0, p1);
  um2::AxisAlignedBox<D> const box3 = box + box2;
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(box3.minima()[i], static_cast<Float>(i), eps);
    ASSERT_NEAR(box3.maxima()[i], static_cast<Float>(i + 2), eps);
  }
}

template <Int D>
HOSTDEV
TEST_CASE(bounding_box)
{
  // boundingBox(Point points[N])
  um2::Point<D> points[2 * D];
  for (Int i = 0; i < D; ++i) {
    um2::Point<D> p_right;
    um2::Point<D> p_left;
    for (Int j = 0; j < D; ++j) {
      p_right[j] = static_cast<Float>(i + 1);
      p_left[j] = -static_cast<Float>(i + 1);
    }
    points[static_cast<size_t>(2 * i + 0)] = p_right;
    points[static_cast<size_t>(2 * i + 1)] = p_left;
  }
  um2::AxisAlignedBox<D> const box4 = um2::boundingBox<D>(points, 2 * D);
  for (Int i = 0; i < D; ++i) {
    ASSERT_NEAR(box4.minima()[i], -static_cast<Float>(D), eps);
    ASSERT_NEAR(box4.maxima()[i], static_cast<Float>(D), eps);
  }
}

TEST_CASE(bounding_box_vector)
{
  // boundingBox(Vector<Point> points)
  Int const n = 20;
  um2::Vector<um2::Point2> points(n);
  for (Int i = 0; i < n; ++i) {
    points[i][0] = static_cast<Float>(i) / 10;
    points[i][1] = static_cast<Float>(i) / 5;
  }
  auto const box = um2::boundingBox<2>(points.begin(), points.end());
  ASSERT_NEAR(box.minima()[0], static_cast<Float>(0), eps);
  ASSERT_NEAR(box.maxima()[0], static_cast<Float>(n - 1) / 10, eps);
  ASSERT_NEAR(box.minima()[1], static_cast<Float>(0), eps);
  ASSERT_NEAR(box.maxima()[1], static_cast<Float>(n - 1) / 5, eps);
}

HOSTDEV
TEST_CASE(intersect_ray)
{
  um2::AxisAlignedBox<2> const box = makeBox<2>();
  // up
  auto const ray0 = um2::Ray2(um2::Point2(half, static_cast<Float>(0)), um2::Point2(0, 1));
  auto const intersection0 = box.intersect(ray0);
  ASSERT_NEAR(intersection0[0], static_cast<Float>(1), eps);
  ASSERT_NEAR(intersection0[1], static_cast<Float>(2), eps);

  auto const ray0_miss = um2::Ray2(um2::Point2(20, 0), um2::Point2(0, -1));
  auto const intersection0_miss = box.intersect(ray0_miss);
  ASSERT(intersection0_miss[0] < 0 );
  ASSERT(intersection0_miss[1] < 0 );

  // right
  auto const ray1 = um2::Ray<2>(um2::Point2(-half, half * 3), um2::Point2(1, 0));
  auto const intersection1 = box.intersect(ray1);
  ASSERT_NEAR(intersection1[0], half, eps);
  ASSERT_NEAR(intersection1[1], half * 3, eps);

  auto const ray1_miss = um2::Ray<2>(um2::Point2(0, 20), um2::Point2(-1, 0));
  auto const intersection1_miss = box.intersect(ray1_miss);
  ASSERT(intersection1_miss[0] < 0);
  ASSERT(intersection1_miss[1] < 0);

  // 45 degrees
  auto const ray2 = um2::Ray<2>(um2::Point2(0, 1), um2::Point2(1, 1).normalized());
  auto const intersection2 = box.intersect(ray2);
  ASSERT_NEAR(intersection2[0], static_cast<Float>(0), eps);
  ASSERT_NEAR(intersection2[1], um2::sqrt(static_cast<Float>(2)), eps);

  auto const ray2_miss = um2::Ray<2>(um2::Point2(0, 20), um2::Point2(-1, -1).normalized());
  auto const intersection2_miss = box.intersect(ray2_miss);
  ASSERT(intersection2_miss[0] < 0);
  ASSERT(intersection2_miss[1] < 0);

  auto const ray3 =
      um2::Ray<2>(um2::Point2(half, half * 3), um2::Point2(1, 1).normalized());
  auto const intersection3 = box.intersect(ray3);
  ASSERT_NEAR(intersection3[0], static_cast<Float>(0), eps);
  ASSERT_NEAR(intersection3[1], um2::sqrt(static_cast<Float>(2)) / 2, eps);

  auto const inv_dir = ray3.inverseDirection();
  auto const intersection4 = box.intersect(ray3, inv_dir);
  ASSERT_NEAR(intersection4[0], static_cast<Float>(0), eps);
  ASSERT_NEAR(intersection4[1], um2::sqrt(static_cast<Float>(2)) / 2, eps);
}

HOSTDEV
TEST_CASE(scale)
{
  um2::AxisAlignedBox<2> box(um2::Point2(0, 0), um2::Point2(1, 1));
  box.scale(2);
  ASSERT_NEAR(box.minima(0), castIfNot<Float>(-0.5), eps);
  ASSERT_NEAR(box.minima(1), castIfNot<Float>(-0.5), eps);
  ASSERT_NEAR(box.maxima(0), castIfNot<Float>(1.5), eps);
  ASSERT_NEAR(box.maxima(1), castIfNot<Float>(1.5), eps);
  ASSERT_NEAR(box.extents(0) * box.extents(1), castIfNot<Float>(4), eps);
  box.scale(castIfNot<Float>(0.5));
  ASSERT_NEAR(box.minima(0), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.minima(1), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.maxima(0), castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.maxima(1), castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.extents(0) * box.extents(1), castIfNot<Float>(1), eps);
}

HOSTDEV
TEST_CASE(intersects_box)
{
  // Simple intersection
  um2::AxisAlignedBox<2> a(um2::Point2(0, 0), um2::Point2(2, 2)); 
  um2::AxisAlignedBox<2> const b(um2::Point2(1, 1), um2::Point2(3, 3));
  ASSERT(a.intersects(b));
  ASSERT(b.intersects(a));
  a += um2::Point2(-1, -1);
  a += um2::Point2(4, 4);
  // a encompasses b
  ASSERT(a.intersects(b));
  ASSERT(b.intersects(a));
  um2::AxisAlignedBox<2> const c(um2::Point2(1, 4), um2::Point2(3, 5));
  // c is directly above b
  ASSERT(!b.intersects(c));
  ASSERT(!c.intersects(b));
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)

#if UM2_USE_CUDA
template <Int D>
MAKE_CUDA_KERNEL(accessors, D);

template <Int D>
MAKE_CUDA_KERNEL(extents, D);

template <Int D>
MAKE_CUDA_KERNEL(centroid, D);

template <Int D>
MAKE_CUDA_KERNEL(contains, D);

template <Int D>
MAKE_CUDA_KERNEL(is_approx, D);

template <Int D>
MAKE_CUDA_KERNEL(operator_plus, D);

template <Int D>
MAKE_CUDA_KERNEL(bounding_box, D);

MAKE_CUDA_KERNEL(intersect_ray);
#endif

template <Int D>
TEST_SUITE(aabb)
{
  TEST_HOSTDEV(extents, D);
  TEST_HOSTDEV(centroid, D);
  TEST_HOSTDEV(contains, D);
  TEST_HOSTDEV(is_approx, D);
  TEST_HOSTDEV(operator_plus, D);
  TEST_HOSTDEV(bounding_box, D);
  if constexpr (D == 2) {
    TEST(bounding_box_vector);
    TEST_HOSTDEV(intersect_ray);
    TEST_HOSTDEV(scale);
    TEST_HOSTDEV(intersects_box);
  }
}

auto
main() -> int
{
  RUN_SUITE(aabb<1>);
  RUN_SUITE(aabb<2>);
  RUN_SUITE(aabb<3>);
  return 0;
}
