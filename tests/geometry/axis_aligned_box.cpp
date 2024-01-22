#include <um2/common/log.hpp>
#include <um2/geometry/axis_aligned_box.hpp>

#include "../test_macros.hpp"

F constexpr eps = um2::eps_distance;
F constexpr half = static_cast<F>(1) / static_cast<F>(2);

// Compiler complains when making this a static_assert, so we silence the warning
// NOLINTBEGIN(cert-dcl03-c,misc-static-assert)

template <Size D>
HOSTDEV constexpr auto
makeBox() -> um2::AxisAlignedBox<D>
{
  um2::Point<D> minima;
  um2::Point<D> maxima;
  for (Size i = 0; i < D; ++i) {
    minima[i] = static_cast<F>(i);
    maxima[i] = static_cast<F>(i + 1);
  }
  return {minima, maxima};
}

template <Size D>
HOSTDEV
TEST_CASE(lengths)
{
  um2::AxisAlignedBox<D> const box = makeBox<D>();
  if constexpr (D >= 1) {
    ASSERT_NEAR(box.width(), 1, eps);
  }
  if constexpr (D >= 2) {
    ASSERT_NEAR(box.height(), 1, eps);
  }
  if constexpr (D >= 3) {
    ASSERT_NEAR(box.depth(), 1, eps);
  }
}

template <Size D>
HOSTDEV
TEST_CASE(centroid)
{
  um2::AxisAlignedBox<D> const box = makeBox<D>();
  um2::Point<D> p = box.centroid();
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(p[i], static_cast<F>(i) + half, eps);
  }
}

template <Size D>
HOSTDEV
TEST_CASE(contains)
{
  um2::AxisAlignedBox<D> const box = makeBox<D>();
  um2::Point<D> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = static_cast<F>(i);
  }
  ASSERT(box.contains(p));
  p[0] += half * eps;
  ASSERT(box.contains(p));
  p[0] -= static_cast<F>(5) * eps;
  ASSERT(!box.contains(p));
}

template <Size D>
HOSTDEV
TEST_CASE(is_approx)
{
  um2::AxisAlignedBox<D> const box1 = makeBox<D>();
  um2::AxisAlignedBox<D> box2 = makeBox<D>();
  ASSERT(isApprox(box1, box2));
  um2::Point<D> p = box2.minima();
  for (Size i = 0; i < D; ++i) {
    p[i] = p[i] - eps / static_cast<F>(10);
  }
  box2 += p;
  ASSERT(isApprox(box1, box2));
  box2 += 10 * p;
  ASSERT(!isApprox(box1, box2));
}

template <Size D>
HOSTDEV
TEST_CASE(operator_plus)
{
  // (AxisAlignedBox, AxisAlignedBox)
  um2::AxisAlignedBox<D> const box = makeBox<D>();
  um2::Point<D> p0 = box.minima();
  um2::Point<D> p1 = box.maxima();
  for (Size i = 0; i < D; ++i) {
    p0[i] += static_cast<F>(1);
    p1[i] += static_cast<F>(1);
  }
  um2::AxisAlignedBox<D> const box2(p0, p1);
  um2::AxisAlignedBox<D> const box3 = box + box2;
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(box3.minima()[i], static_cast<F>(i), eps);
    ASSERT_NEAR(box3.maxima()[i], static_cast<F>(i + 2), eps); 
  }
}

template <Size D>
HOSTDEV
TEST_CASE(bounding_box)
{
  // boundingBox(Point points[N])
  um2::Point<D> points[2 * D];
  for (Size i = 0; i < D; ++i) {
    um2::Point<D> p_right;
    um2::Point<D> p_left;
    for (Size j = 0; j < D; ++j) {
      p_right[j] = static_cast<F>(i + 1);
      p_left[j] = -static_cast<F>(i + 1);
    }
    points[static_cast<size_t>(2 * i + 0)] = p_right;
    points[static_cast<size_t>(2 * i + 1)] = p_left;
  }
  um2::AxisAlignedBox<D> const box4 = um2::boundingBox(points, 2 * D);
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(box4.minima()[i], -static_cast<F>(D), eps); 
    ASSERT_NEAR(box4.maxima()[i], static_cast<F>(D), eps);
  }
}

TEST_CASE(bounding_box_vector)
{
  // boundingBox(Vector<Point> points)
  Size const n = 20;
  um2::Vector<um2::Point2> points(n);
  for (Size i = 0; i < n; ++i) {
    points[i][0] = static_cast<F>(i) / 10;
    points[i][1] = static_cast<F>(i) / 5;
  }
  auto const box = um2::boundingBox(points);
  ASSERT_NEAR(box.minima()[0], static_cast<F>(0), eps);
  ASSERT_NEAR(box.maxima()[0], static_cast<F>(n - 1) / 10, eps);
  ASSERT_NEAR(box.minima()[1], static_cast<F>(0), eps);
  ASSERT_NEAR(box.maxima()[1], static_cast<F>(n - 1) / 5, eps);
}

HOSTDEV
TEST_CASE(intersect_ray)
{
  um2::AxisAlignedBox<2> const box = makeBox<2>();
  auto const ray0 = um2::Ray2(um2::Point2(half, static_cast<F>(0)),
                                   um2::Vec2<F>(0, 1));

  auto const intersection0 = um2::intersect(ray0, box);
  ASSERT_NEAR(intersection0[0], static_cast<F>(1), eps);
  ASSERT_NEAR(intersection0[1], static_cast<F>(2), eps);

  auto const ray1 = um2::Ray<2>(
      um2::Point2(-half, half * 3), um2::Vec2<F>(1, 0));
  auto const intersection1 = um2::intersect(ray1, box);
  ASSERT_NEAR(intersection1[0], half, eps);
  ASSERT_NEAR(intersection1[1], half * 3, eps);

  auto const ray2 = um2::Ray<2>(um2::Point2(0, 1), um2::Vec2<F>(1, 1).normalized());
  auto const intersection2 = um2::intersect(ray2, box);
  ASSERT_NEAR(intersection2[0], static_cast<F>(0), eps);
  ASSERT_NEAR(intersection2[1], um2::sqrt(static_cast<F>(2)), eps);

  auto const ray3 =
      um2::Ray<2>(um2::Point2(half, half * 3),
                     um2::Vec2<F>(1, 1).normalized());
  auto const intersection3 = um2::intersect(ray3, box);
  ASSERT_NEAR(intersection3[0], static_cast<F>(0), eps);
  ASSERT_NEAR(intersection3[1], um2::sqrt(static_cast<F>(2)) / 2, eps);
}

// NOLINTEND(cert-dcl03-c,misc-static-assert)

#if UM2_USE_CUDA
template <Size D>
MAKE_CUDA_KERNEL(accessors, D);

template <Size D>
MAKE_CUDA_KERNEL(lengths, D);

template <Size D>
MAKE_CUDA_KERNEL(centroid, D);

template <Size D>
MAKE_CUDA_KERNEL(contains, D);

template <Size D>
MAKE_CUDA_KERNEL(is_approx, D);

template <Size D>
MAKE_CUDA_KERNEL(operator_plus, D);

template <Size D>
MAKE_CUDA_KERNEL(bounding_box, D);

MAKE_CUDA_KERNEL(intersect_ray);
#endif

template <Size D>
TEST_SUITE(aabb)
{
  TEST_HOSTDEV(lengths, 1, 1, D);
  TEST_HOSTDEV(centroid, 1, 1, D);
  TEST_HOSTDEV(contains, 1, 1, D);
  TEST_HOSTDEV(is_approx, 1, 1, D);
  TEST_HOSTDEV(operator_plus, 1, 1, D);
  TEST_HOSTDEV(bounding_box, 1, 1, D);
  if constexpr (D == 2) {
    TEST(bounding_box_vector);
    TEST_HOSTDEV(intersect_ray);
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
