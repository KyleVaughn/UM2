#include <um2/common/log.hpp>
#include <um2/geometry/axis_aligned_box.hpp>

#include "../test_macros.hpp"

template <Size D, std::floating_point T>
HOSTDEV constexpr auto
makeBox() -> um2::AxisAlignedBox<D, T>
{
  um2::Point<D, T> minima;
  um2::Point<D, T> maxima;
  for (Size i = 0; i < D; ++i) {
    minima[i] = static_cast<T>(i);
    maxima[i] = static_cast<T>(i + 1);
  }
  return {minima, maxima};
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(lengths)
{
  um2::AxisAlignedBox<D, T> const box = makeBox<D, T>();
  if constexpr (D >= 1) {
    ASSERT_NEAR(box.width(), 1, static_cast<T>(1e-5));
  }
  if constexpr (D >= 2) {
    ASSERT_NEAR(box.height(), 1, static_cast<T>(1e-5));
  }
  if constexpr (D >= 3) {
    ASSERT_NEAR(box.depth(), 1, static_cast<T>(1e-5));
  }
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(centroid)
{
  um2::AxisAlignedBox<D, T> const box = makeBox<D, T>();
  um2::Point<D, T> p = box.centroid();
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(p[i], static_cast<T>(i) + static_cast<T>(0.5), static_cast<T>(1e-6));
  }
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(contains)
{
  um2::AxisAlignedBox<D, T> const box = makeBox<D, T>();
  um2::Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = static_cast<T>(i);
  }
  ASSERT(box.contains(p));
  p[0] += static_cast<T>(0.5) * um2::eps_distance<T>;
  ASSERT(box.contains(p));
  p[0] -= static_cast<T>(2.5) * um2::eps_distance<T>;
  ASSERT(!box.contains(p));
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(is_approx)
{
  um2::AxisAlignedBox<D, T> const box1 = makeBox<D, T>();
  um2::AxisAlignedBox<D, T> box2 = makeBox<D, T>();
  ASSERT(isApprox(box1, box2));
  um2::Point<D, T> p = box2.minima();
  for (Size i = 0; i < D; ++i) {
    p[i] = p[i] - um2::eps_distance<T> / static_cast<T>(10);
  }
  box2 += p;
  ASSERT(isApprox(box1, box2));
  box2 += 10 * p;
  ASSERT(!isApprox(box1, box2));
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(operator_plus)
{
  // (AxisAlignedBox, AxisAlignedBox)
  um2::AxisAlignedBox<D, T> const box = makeBox<D, T>();
  um2::Point<D, T> p0 = box.minima();
  um2::Point<D, T> p1 = box.maxima();
  for (Size i = 0; i < D; ++i) {
    p0[i] += static_cast<T>(1);
    p1[i] += static_cast<T>(1);
  }
  um2::AxisAlignedBox<D, T> const box2(p0, p1);
  um2::AxisAlignedBox<D, T> const box3 = box + box2;
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(box3.minima()[i], static_cast<T>(i), static_cast<T>(1e-6));
    ASSERT_NEAR(box3.maxima()[i], static_cast<T>(i + 2), static_cast<T>(1e-6));
  }
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(bounding_box)
{
  // boundingBox(Point points[N])
  um2::Point<D, T> points[2 * D];
  for (Size i = 0; i < D; ++i) {
    um2::Point<D, T> p_right;
    um2::Point<D, T> p_left;
    for (Size j = 0; j < D; ++j) {
      p_right[j] = static_cast<T>(i + 1);
      p_left[j] = -static_cast<T>(i + 1);
    }
    points[static_cast<size_t>(2 * i + 0)] = p_right;
    points[static_cast<size_t>(2 * i + 1)] = p_left;
  }
  um2::AxisAlignedBox<D, T> const box4 = um2::boundingBox(points, 2 * D);
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(box4.minima()[i], -static_cast<T>(D), static_cast<T>(1e-6));
    ASSERT_NEAR(box4.maxima()[i], static_cast<T>(D), static_cast<T>(1e-6));
  }
}

template <std::floating_point T>
TEST_CASE(bounding_box_vector)
{
  // boundingBox(Vector<Point> points)
  Size const n = 20;
  um2::Vector<um2::Point2<T>> points(n);
  for (Size i = 0; i < n; ++i) {
    points[i][0] = static_cast<T>(0.1) * static_cast<T>(i);
    points[i][1] = static_cast<T>(0.2) * static_cast<T>(i);
  }
  auto const box = um2::boundingBox(points);
  ASSERT_NEAR(box.minima()[0], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.maxima()[0], static_cast<T>(0.1 * (n - 1)), static_cast<T>(1e-6));
  ASSERT_NEAR(box.minima()[1], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.maxima()[1], static_cast<T>(0.2 * (n - 1)), static_cast<T>(1e-6));
}

template <std::floating_point T>
HOSTDEV
TEST_CASE(intersect_ray)
{
  um2::AxisAlignedBox<2, T> const box = makeBox<2, T>();
  auto const ray0 = um2::Ray<2, T>(um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0)),
                                   um2::Vec2<T>(0, 1));

  auto const intersection0 = um2::intersect(ray0, box);
  ASSERT_NEAR(intersection0[0], static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(intersection0[1], static_cast<T>(2), static_cast<T>(1e-6));

  auto const ray1 = um2::Ray<2, T>(
      um2::Point2<T>(static_cast<T>(-0.5), static_cast<T>(1.5)), um2::Vec2<T>(1, 0));
  auto const intersection1 = um2::intersect(ray1, box);
  ASSERT_NEAR(intersection1[0], static_cast<T>(0.5), static_cast<T>(1e-6));
  ASSERT_NEAR(intersection1[1], static_cast<T>(1.5), static_cast<T>(1e-6));

  auto const ray2 = um2::Ray<2, T>(um2::Point2<T>(0, 1), um2::Vec2<T>(1, 1).normalized());
  auto const intersection2 = um2::intersect(ray2, box);
  ASSERT_NEAR(intersection2[0], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(intersection2[1], um2::sqrt(static_cast<T>(2)), static_cast<T>(1e-6));

  auto const ray3 =
      um2::Ray<2, T>(um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(1.5)),
                     um2::Vec2<T>(1, 1).normalized());
  auto const intersection3 = um2::intersect(ray3, box);
  ASSERT_NEAR(intersection3[0], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(intersection3[1], um2::sqrt(static_cast<T>(2)) / 2, static_cast<T>(1e-6));
}

#if UM2_USE_CUDA
template <Size D, std::floating_point T>
MAKE_CUDA_KERNEL(accessors, D, T);

template <Size D, std::floating_point T>
MAKE_CUDA_KERNEL(lengths, D, T);

template <Size D, std::floating_point T>
MAKE_CUDA_KERNEL(centroid, D, T);

template <Size D, std::floating_point T>
MAKE_CUDA_KERNEL(contains, D, T);

template <Size D, std::floating_point T>
MAKE_CUDA_KERNEL(is_approx, D, T);

template <Size D, std::floating_point T>
MAKE_CUDA_KERNEL(operator_plus, D, T);

template <Size D, std::floating_point T>
MAKE_CUDA_KERNEL(bounding_box, D, T);

template <std::floating_point T>
MAKE_CUDA_KERNEL(intersect_ray, T);
#endif

template <Size D, std::floating_point T>
TEST_SUITE(aabb)
{
  TEST_HOSTDEV(lengths, 1, 1, D, T);
  TEST_HOSTDEV(centroid, 1, 1, D, T);
  TEST_HOSTDEV(contains, 1, 1, D, T);
  TEST_HOSTDEV(is_approx, 1, 1, D, T);
  TEST_HOSTDEV(operator_plus, 1, 1, D, T);
  TEST_HOSTDEV(bounding_box, 1, 1, D, T);
  if constexpr (D == 2) {
    TEST((bounding_box_vector<T>));
    TEST_HOSTDEV(intersect_ray, 1, 1, T);
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
