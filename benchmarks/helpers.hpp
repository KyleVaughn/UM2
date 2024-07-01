#include <um2/common/logger.hpp>
#include <um2/stdlib/vector.hpp>

#include <random>

template <class T>
auto
randomFloat() -> T
{
  // Random number in range [0, 1]
  // NOLINTNEXTLINE(cert*) justification: Don't care about cryptographic randomness.
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(0, 1);
  return dist(rng);
}

template <class T>
auto
randomInt() -> T
{
  // Random number in range [0, 100]
  // NOLINTNEXTLINE(cert*) justification: Don't care about cryptographic randomness.
  static std::default_random_engine rng;
  static std::uniform_int_distribution<T> dist(0, 100);
  return dist(rng);
}

template <class T>
auto
makeVectorOfRandomFloats(Int size, T lo = 0, T hi = 1) -> um2::Vector<T>
{
  um2::Vector<T> v(size);
  for (auto & x : v) {
    x = lo + randomFloat<T>() * (hi - lo);
  }
  return v;
}

template <class T>
auto
makeVectorOfRandomInts(Int size) -> um2::Vector<T>
{
  um2::Vector<T> v(size);
  for (auto & x : v) {
    x = randomInt<T>();
  }
  return v;
}

#if UM2_USE_CUDA
template <class T>
void
transferToDevice(T ** d_v, um2::Vector<T> const & v)
{
  size_t const size_in_bytes = static_cast<size_t>(v.size()) * sizeof(T);
  cudaMalloc(d_v, size_in_bytes);
  cudaMemcpy(*d_v, v.data(), size_in_bytes, cudaMemcpyHostToDevice);
}

template <class T>
void
transferFromDevice(um2::Vector<T> & v, T const * d_v)
{
  if (v.empty()) {
    um2::logger::error("Cannot transfer from device to host, vector is empty.");
    return;
  }
  size_t const size_in_bytes = static_cast<size_t>(v.size()) * sizeof(T);
  cudaMemcpy(v.data(), d_v, size_in_bytes, cudaMemcpyDeviceToHost);
}

#endif
