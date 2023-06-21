#include "../test_framework.hpp"
#include <um2/geometry/axis_aligned_box.hpp>

template <len_t D, std::floating_point T>
UM2_HOSTDEV static constexpr auto
makeBox() -> um2::AABox<D, T>
{
  um2::AABox<D, T> box;
  for (len_t i = 0; i < D; ++i) {
    box.minima[i] = static_cast<T>(i);
    box.maxima[i] = static_cast<T>(i + 1);
  }
  return box;
}

template <len_t D, std::floating_point T>
UM2_HOSTDEV
TEST_CASE(accessors)
{
  um2::AABox<D, T> box = makeBox<D, T>();
  if constexpr (D >= 1) {
    EXPECT_NEAR(box.xmin(), 0, 1e-6);
    EXPECT_NEAR(box.xmax(), 1, 1e-6);
  }
  if constexpr (D >= 2) {
    EXPECT_NEAR(box.ymin(), 1, 1e-6);
    EXPECT_NEAR(box.ymax(), 2, 1e-6);
  }
  if constexpr (D >= 3) {
    EXPECT_NEAR(box.zmin(), 2, 1e-6);
    EXPECT_NEAR(box.zmax(), 3, 1e-6);
  }
}

template <len_t D, std::floating_point T>
UM2_HOSTDEV
TEST_CASE(lengths)
{
  um2::AABox<D, T> box = makeBox<D, T>();
  if constexpr (D >= 1) {
    EXPECT_NEAR(box.width(), 1, 1e-5);
  }
  if constexpr (D >= 2) {
    EXPECT_NEAR(box.height(), 1, 1e-5);
  }
  if constexpr (D >= 3) {
    EXPECT_NEAR(box.depth(), 1, 1e-5);
  }
}

template <len_t D, std::floating_point T>
UM2_HOSTDEV
TEST_CASE(centroid)
{
  um2::AABox<D, T> box = makeBox<D, T>();
  um2::Point<D, T> p = box.centroid();
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(p[i], static_cast<T>(i) + static_cast<T>(0.5), 1e-6);
  }
}

template <len_t D, std::floating_point T>
UM2_HOSTDEV
TEST_CASE(contains)
{
  um2::AABox<D, T> box = makeBox<D, T>();
  um2::Point<D, T> p;
  for (len_t i = 0; i < D; ++i) {
    p[i] = static_cast<T>(i);
  }
  EXPECT_TRUE(box.contains(p));
  p[0] += static_cast<T>(0.5) * um2::epsilonDistance<T>();
  EXPECT_TRUE(box.contains(p));
  p[0] -= static_cast<T>(2.5) * um2::epsilonDistance<T>();
  EXPECT_FALSE(box.contains(p));
}

template <len_t D, std::floating_point T>
UM2_HOSTDEV
TEST_CASE(is_approx)
{
  um2::AABox<D, T> box1 = makeBox<D, T>();
  um2::AABox<D, T> box2 = makeBox<D, T>();
  EXPECT_TRUE(isApprox(box1, box2));
  box2.maxima[0] += um2::epsilonDistance<T>() / 2;
  EXPECT_TRUE(isApprox(box1, box2));
  box2.maxima[0] += um2::epsilonDistance<T>();
  EXPECT_FALSE(isApprox(box1, box2));
}

template <len_t D, std::floating_point T>
UM2_HOSTDEV
TEST_CASE(bounding_box)
{
  // boundingBox(AABox, AABox)
  um2::AABox<D, T> box = makeBox<D, T>();
  um2::AABox<D, T> box2 = makeBox<D, T>();
  for (len_t i = 0; i < D; ++i) {
    box2.minima[i] += static_cast<T>(1);
    box2.maxima[i] += static_cast<T>(1);
  }
  um2::AABox<D, T> box3 = boundingBox(box, box2);
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(box3.minima[i], i, 1e-6);
    EXPECT_NEAR(box3.maxima[i], i + 2, 1e-6);
  }
  // boundingBox(Point points[N])
  um2::Point<D, T> points[2 * D];
  for (len_t i = 0; i < D; ++i) {
    um2::Point<D, T> p_right;
    um2::Point<D, T> p_left;
    for (len_t j = 0; j < D; ++j) {
      p_right[j] = static_cast<T>(i + 1);
      p_left[j] = -static_cast<T>(i + 1);
    }
    points[static_cast<size_t>(2 * i + 0)] = p_right;
    points[static_cast<size_t>(2 * i + 1)] = p_left;
  }
  um2::AABox<D, T> box4 = um2::boundingBox(points);
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(box4.minima[i], -static_cast<T>(D), 1e-6);
    EXPECT_NEAR(box4.maxima[i], static_cast<T>(D), 1e-6);
  }
  // boundingBox(Vector<Point> points)
  um2::Vector<um2::Point<D, T>> points2(2 * D);
  for (len_t i = 0; i < D; ++i) {
    um2::Point<D, T> p_right;
    um2::Point<D, T> p_left;
    for (len_t j = 0; j < D; ++j) {
      p_right[j] = static_cast<T>(i + 1);
      p_left[j] = -static_cast<T>(i + 1);
    }
    points2[2 * i + 0] = p_right;
    points2[2 * i + 1] = p_left;
  }
  um2::AABox<D, T> box5 = um2::boundingBox(points2);
  for (len_t i = 0; i < D; ++i) {
    EXPECT_NEAR(box5.minima[i], -static_cast<T>(D), 1e-6);
    EXPECT_NEAR(box5.maxima[i], static_cast<T>(D), 1e-6);
  }
}

#if UM2_ENABLE_CUDA
template <len_t D, std::floating_point T>
MAKE_CUDA_KERNEL(accessors, D, T);

template <len_t D, std::floating_point T>
MAKE_CUDA_KERNEL(lengths, D, T);

template <len_t D, std::floating_point T>
MAKE_CUDA_KERNEL(centroid, D, T);

template <len_t D, std::floating_point T>
MAKE_CUDA_KERNEL(contains, D, T);

template <len_t D, std::floating_point T>
MAKE_CUDA_KERNEL(is_approx, D, T);

template <len_t D, std::floating_point T>
MAKE_CUDA_KERNEL(bounding_box, D, T);
#endif

template <len_t D, std::floating_point T>
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
  RUN_TESTS((aabb<1, float>));
  RUN_TESTS((aabb<2, float>));
  RUN_TESTS((aabb<3, float>));
  RUN_TESTS((aabb<1, double>));
  RUN_TESTS((aabb<2, double>));
  RUN_TESTS((aabb<3, double>));
  return 0;
}
