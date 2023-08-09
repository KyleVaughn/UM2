#include <benchmark/benchmark.h>

#include <um2/common/Log.hpp>
#include <um2/geometry/Point.hpp>
#include <um2/geometry/Triangle.hpp>
#include <um2/stdlib/Vector.hpp>

#include <algorithm>
#if UM2_ENABLE_OPENMP
#  include <parallel/algorithm>
#endif
#include <random>

template <typename T, int lo, int hi>
auto
randomFloat() -> T
{
  // NOLINTNEXTLINE
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(lo, hi);
  return dist(rng);
}

template <typename T, int lo, int hi>
auto
makeVectorOfRandomFloats(Size size) -> um2::Vector<T>
{
  um2::Vector<T> v(size);
  std::generate(v.begin(), v.end(), randomFloat<T, lo, hi>);
  return v;
}

template <Size D, typename T, int lo, int hi>
auto
randomPoint() -> um2::Point<D, T>
{
  um2::Point<D, T> p;
  std::generate(p.begin(), p.end(), randomFloat<T, lo, hi>);
  return p;
}

template <Size D, typename T, int lo, int hi>
auto
makeVectorOfRandomPoints(Size size) -> um2::Vector<um2::Point<D, T>>
{
  um2::Vector<um2::Point<D, T>> v(size);
  std::generate(v.begin(), v.end(), randomPoint<D, T, lo, hi>);
  return v;
}

template <typename T, int lo, int hi>
auto
makeVectorOfRandomTriangles(Size size) -> um2::Vector<um2::Triangle2<T>>
{
  um2::Vector<um2::Triangle2<T>> v(size);
  for (auto & t : v) {
    t[0] = randomPoint<2, T, lo, hi>();
    t[1] = randomPoint<2, T, lo, hi>();
    // We require that the third point is CCW from the first two.
    t[2] = randomPoint<2, T, lo, hi>();
    while (!um2::areCCW(t[0], t[1], t[2])) {
      t[2] = randomPoint<2, T, lo, hi>();
    }
  }
  return v;
}

#if UM2_ENABLE_CUDA
template <typename T>
void
transferToDevice(T ** d_v, um2::Vector<T> const & v)
{
  size_t const size_in_bytes = static_cast<size_t>(v.size()) * sizeof(T);
  cudaMalloc(d_v, size_in_bytes);
  cudaMemcpy(*d_v, v.data(), size_in_bytes, cudaMemcpyHostToDevice);
}

template <typename T>
void
transferFromDevice(um2::Vector<T> & v, T const * d_v)
{
  if (v.empty()) {
    um2::Log::error("Cannot transfer from device to host, vector is empty.");
    return;
  }
  size_t const size_in_bytes = static_cast<size_t>(v.size()) * sizeof(T);
  cudaMemcpy(v.data(), d_v, size_in_bytes, cudaMemcpyDeviceToHost);
}

#endif
