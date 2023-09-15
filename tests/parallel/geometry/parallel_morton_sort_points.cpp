#include <um2/parallel/geometry/morton_sort_points.hpp>
#include <um2/stdlib/Vector.hpp>

#include <random>

#include "../../test_macros.hpp"

template <std::unsigned_integral U, std::floating_point T>
TEST_CASE(mortonSort2D)
{
  um2::Vector<um2::Point2<T>> points(16);
  for (Size i = 0; i < 4; ++i) {
    for (Size j = 0; j < 4; ++j) {
      points[i * 4 + j] = um2::Point2<T>(static_cast<T>(i), static_cast<T>(j)) / 3;
    }
  }
  um2::parallel::mortonSort<U>(points.begin(), points.end());
  ASSERT(um2::isApprox(points[0], um2::Point2<T>(0, 0) / 3));
  ASSERT(um2::isApprox(points[1], um2::Point2<T>(1, 0) / 3));
  ASSERT(um2::isApprox(points[2], um2::Point2<T>(0, 1) / 3));
  ASSERT(um2::isApprox(points[3], um2::Point2<T>(1, 1) / 3));
  ASSERT(um2::isApprox(points[4], um2::Point2<T>(2, 0) / 3));
  ASSERT(um2::isApprox(points[5], um2::Point2<T>(3, 0) / 3));
  ASSERT(um2::isApprox(points[6], um2::Point2<T>(2, 1) / 3));
  ASSERT(um2::isApprox(points[7], um2::Point2<T>(3, 1) / 3));
  ASSERT(um2::isApprox(points[8], um2::Point2<T>(0, 2) / 3));
  ASSERT(um2::isApprox(points[9], um2::Point2<T>(1, 2) / 3));
  ASSERT(um2::isApprox(points[10], um2::Point2<T>(0, 3) / 3));
  ASSERT(um2::isApprox(points[11], um2::Point2<T>(1, 3) / 3));
  ASSERT(um2::isApprox(points[12], um2::Point2<T>(2, 2) / 3));
  ASSERT(um2::isApprox(points[13], um2::Point2<T>(3, 2) / 3));
  ASSERT(um2::isApprox(points[14], um2::Point2<T>(2, 3) / 3));
  ASSERT(um2::isApprox(points[15], um2::Point2<T>(3, 3) / 3));
}

template <std::unsigned_integral U, std::floating_point T>
TEST_CASE(mortonSort3D)
{
  um2::Vector<um2::Point3<T>> points(64);
  for (Size i = 0; i < 4; ++i) {
    for (Size j = 0; j < 4; ++j) {
      for (Size k = 0; k < 4; ++k) {
        points[i * 16 + j * 4 + k] =
            um2::Point3<T>(static_cast<T>(i), static_cast<T>(j), static_cast<T>(k)) / 3;
      }
    }
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(points.begin(), points.end(), g);
  um2::parallel::mortonSort<U>(points.begin(), points.end());
  ASSERT(um2::isApprox(points[0], um2::Point3<T>(0, 0, 0) / 3));
  ASSERT(um2::isApprox(points[1], um2::Point3<T>(1, 0, 0) / 3));
  ASSERT(um2::isApprox(points[2], um2::Point3<T>(0, 1, 0) / 3));
  ASSERT(um2::isApprox(points[3], um2::Point3<T>(1, 1, 0) / 3));
  ASSERT(um2::isApprox(points[4], um2::Point3<T>(0, 0, 1) / 3));
  ASSERT(um2::isApprox(points[5], um2::Point3<T>(1, 0, 1) / 3));
  ASSERT(um2::isApprox(points[6], um2::Point3<T>(0, 1, 1) / 3));
  ASSERT(um2::isApprox(points[7], um2::Point3<T>(1, 1, 1) / 3));
  ASSERT(um2::isApprox(points[8], um2::Point3<T>(2, 0, 0) / 3));
  ASSERT(um2::isApprox(points[9], um2::Point3<T>(3, 0, 0) / 3));
  ASSERT(um2::isApprox(points[10], um2::Point3<T>(2, 1, 0) / 3));
  ASSERT(um2::isApprox(points[11], um2::Point3<T>(3, 1, 0) / 3));
  ASSERT(um2::isApprox(points[12], um2::Point3<T>(2, 0, 1) / 3));
  ASSERT(um2::isApprox(points[13], um2::Point3<T>(3, 0, 1) / 3));
  ASSERT(um2::isApprox(points[14], um2::Point3<T>(2, 1, 1) / 3));
  ASSERT(um2::isApprox(points[15], um2::Point3<T>(3, 1, 1) / 3));
}

#if UM2_USE_CUDA
template <std::unsigned_integral U, std::floating_point T>
TEST_CASE(deviceMortonSort)
{
  um2::Vector<um2::Point2<T>> points(16);
  um2::Vector<um2::Point2<T>> after(16);
  for (Size i = 0; i < 4; ++i) {
    for (Size j = 0; j < 4; ++j) {
      points[i * 4 + j] = um2::Point2<T>(static_cast<T>(i), static_cast<T>(j)) / 3;
    }
  }
  um2::Point2<T> * d_points;
  size_t const size_in_bytes =
      sizeof(um2::Point2<T>) * static_cast<size_t>(points.size());
  cudaMalloc(&d_points, size_in_bytes);
  cudaMemcpy(d_points, points.data(), size_in_bytes, cudaMemcpyHostToDevice);
  um2::parallel::deviceMortonSort<U>(d_points, d_points + points.size());
  cudaMemcpy(after.data(), d_points, size_in_bytes, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  ASSERT(err == cudaSuccess);
  ASSERT(um2::isApprox(after[0], um2::Point2<T>(0, 0) / 3));
  ASSERT(um2::isApprox(after[1], um2::Point2<T>(1, 0) / 3));
  ASSERT(um2::isApprox(after[2], um2::Point2<T>(0, 1) / 3));
  ASSERT(um2::isApprox(after[3], um2::Point2<T>(1, 1) / 3));
  ASSERT(um2::isApprox(after[4], um2::Point2<T>(2, 0) / 3));
  ASSERT(um2::isApprox(after[5], um2::Point2<T>(3, 0) / 3));
  ASSERT(um2::isApprox(after[6], um2::Point2<T>(2, 1) / 3));
  ASSERT(um2::isApprox(after[7], um2::Point2<T>(3, 1) / 3));
  ASSERT(um2::isApprox(after[8], um2::Point2<T>(0, 2) / 3));
  ASSERT(um2::isApprox(after[9], um2::Point2<T>(1, 2) / 3));
  ASSERT(um2::isApprox(after[10], um2::Point2<T>(0, 3) / 3));
  ASSERT(um2::isApprox(after[11], um2::Point2<T>(1, 3) / 3));
  ASSERT(um2::isApprox(after[12], um2::Point2<T>(2, 2) / 3));
  ASSERT(um2::isApprox(after[13], um2::Point2<T>(3, 2) / 3));
  ASSERT(um2::isApprox(after[14], um2::Point2<T>(2, 3) / 3));
  ASSERT(um2::isApprox(after[15], um2::Point2<T>(3, 3) / 3));

  cudaFree(d_points);
}
#endif

template <std::unsigned_integral U, std::floating_point T>
TEST_SUITE(mortonSort)
{
#if UM2_USE_TBB
  TEST((mortonSort2D<U, T>));
  TEST((mortonSort3D<U, T>));
#endif
#if UM2_USE_CUDA
  TEST((deviceMortonSort<U, T>));
#endif
}

auto
main() -> int
{
  RUN_SUITE((mortonSort<uint32_t, float>));
  RUN_SUITE((mortonSort<uint32_t, double>));
  RUN_SUITE((mortonSort<uint64_t, double>));
  return 0;
}
