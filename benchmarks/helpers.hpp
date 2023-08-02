#include <benchmark/benchmark.h>

#include <um2/common/Log.hpp>
#include <um2/common/Vector.hpp>

#include <algorithm>
#if UM2_ENABLE_OPENMP
#  include <parallel/algorithm>
#endif
#include <random>

template <typename T, int Lo, int Hi>
auto
randomFloat() -> T
{
  // NOLINTNEXTLINE
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(Lo, Hi);
  return dist(rng);
}

template <typename T, int Lo, int Hi>
auto
makeVectorOfRandomFloats(Size size) -> um2::Vector<T>
{
  um2::Vector<T> v(size);
  std::generate(v.begin(), v.end(), randomFloat<T, Lo, Hi>);
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
