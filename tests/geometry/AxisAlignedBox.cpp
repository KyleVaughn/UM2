#include <um2/geometry/AxisAlignedBox.hpp>

#include "../test_macros.hpp"

template <Size D, std::floating_point T>
HOSTDEV static constexpr auto
makeBox() -> um2::AxisAlignedBox<D, T>
{
  um2::AxisAlignedBox<D, T> box;
  for (Size i = 0; i < D; ++i) {
    box.minima[i] = static_cast<T>(i);
    box.maxima[i] = static_cast<T>(i + 1);
  }
  return box;
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(accessors)
{
  um2::AxisAlignedBox<D, T> box = makeBox<D, T>();
  if constexpr (D >= 1) {
    ASSERT_NEAR(box.xMin(), 0, static_cast<T>(1e-6));
    ASSERT_NEAR(box.xMax(), 1, static_cast<T>(1e-6));
  }
  if constexpr (D >= 2) {
    ASSERT_NEAR(box.yMin(), 1, static_cast<T>(1e-6));
    ASSERT_NEAR(box.yMax(), 2, static_cast<T>(1e-6));
  }
  if constexpr (D >= 3) {
    ASSERT_NEAR(box.zMin(), 2, static_cast<T>(1e-6));
    ASSERT_NEAR(box.zMax(), 3, static_cast<T>(1e-6));
  }
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(lengths)
{
  um2::AxisAlignedBox<D, T> box = makeBox<D, T>();
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
  um2::AxisAlignedBox<D, T> box = makeBox<D, T>();
  um2::Point<D, T> p = box.centroid();
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(p[i], static_cast<T>(i) + static_cast<T>(0.5), static_cast<T>(1e-6));
  }
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(contains)
{
  um2::AxisAlignedBox<D, T> box = makeBox<D, T>();
  um2::Point<D, T> p;
  for (Size i = 0; i < D; ++i) {
    p[i] = static_cast<T>(i);
  }
  ASSERT(box.contains(p));
  p[0] += static_cast<T>(0.5) * um2::epsilonDistance<T>();
  ASSERT(box.contains(p));
  p[0] -= static_cast<T>(2.5) * um2::epsilonDistance<T>();
  ASSERT(!box.contains(p));
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(is_approx)
{
  um2::AxisAlignedBox<D, T> box1 = makeBox<D, T>();
  um2::AxisAlignedBox<D, T> box2 = makeBox<D, T>();
  ASSERT(isApprox(box1, box2));
  box2.maxima[0] += um2::epsilonDistance<T>() / 2;
  ASSERT(isApprox(box1, box2));
  box2.maxima[0] += um2::epsilonDistance<T>();
  ASSERT(!isApprox(box1, box2));
}

template <Size D, std::floating_point T>
HOSTDEV
TEST_CASE(bounding_box)
{
  // boundingBox(AxisAlignedBox, AxisAlignedBox)
  um2::AxisAlignedBox<D, T> box = makeBox<D, T>();
  um2::AxisAlignedBox<D, T> box2 = makeBox<D, T>();
  for (Size i = 0; i < D; ++i) {
    box2.minima[i] += static_cast<T>(1);
    box2.maxima[i] += static_cast<T>(1);
  }
  um2::AxisAlignedBox<D, T> box3 = boundingBox(box, box2);
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(box3.minima[i], static_cast<T>(i), static_cast<T>(1e-6));
    ASSERT_NEAR(box3.maxima[i], static_cast<T>(i + 2), static_cast<T>(1e-6));
  }
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
  um2::AxisAlignedBox<D, T> box4 = um2::boundingBox(points);
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(box4.minima[i], -static_cast<T>(D), static_cast<T>(1e-6));
    ASSERT_NEAR(box4.maxima[i], static_cast<T>(D), static_cast<T>(1e-6));
  }
  // boundingBox(Vector<Point> points)
  um2::Vector<um2::Point<D, T>> points2(2 * D);
  for (Size i = 0; i < D; ++i) {
    um2::Point<D, T> p_right;
    um2::Point<D, T> p_left;
    for (Size j = 0; j < D; ++j) {
      p_right[j] = static_cast<T>(i + 1);
      p_left[j] = -static_cast<T>(i + 1);
    }
    points2[2 * i + 0] = p_right;
    points2[2 * i + 1] = p_left;
  }
  um2::AxisAlignedBox<D, T> box5 = um2::boundingBox(points2);
  for (Size i = 0; i < D; ++i) {
    ASSERT_NEAR(box5.minima[i], -static_cast<T>(D), static_cast<T>(1e-6));
    ASSERT_NEAR(box5.maxima[i], static_cast<T>(D), static_cast<T>(1e-6));
  }
}

#if UM2_ENABLE_CUDA
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
MAKE_CUDA_KERNEL(bounding_box, D, T);
#endif

template <Size D, std::floating_point T>
TEST_SUITE(aabb)
{
  TEST_HOSTDEV(accessors, 1, 1, D, T);
  TEST_HOSTDEV(lengths, 1, 1, D, T);
  TEST_HOSTDEV(centroid, 1, 1, D, T);
  TEST_HOSTDEV(contains, 1, 1, D, T);
  TEST_HOSTDEV(is_approx, 1, 1, D, T);
  TEST_HOSTDEV(bounding_box, 1, 1, D, T);
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
