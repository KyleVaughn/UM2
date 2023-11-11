#include <benchmark/benchmark.h>

//#include <um2/common/Log.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>
//#include <um2/geometry/Polygon.hpp>
#include <um2/stdlib/vector.hpp>
//
//#include <algorithm>
//#if UM2_USE_TBB
//#  include <execution>
//#endif

#include <iostream>
#include <random>

template <typename T>
auto
randomFloat() -> T
{
  // Random number in range [0, 1]
  // NOLINTNEXTLINE(cert*) justification: Don't care about cryptographic randomness.
  static std::default_random_engine rng;
  static std::uniform_real_distribution<T> dist(0, 1);
  return dist(rng);
}

template <typename T>
auto
randomInt() -> T
{
  // Random number in range [0, 100]
  // NOLINTNEXTLINE(cert*) justification: Don't care about cryptographic randomness.
  static std::default_random_engine rng;
  static std::uniform_int_distribution<T> dist(0, 1000);
  return dist(rng);
}

template <typename T>
auto
makeVectorOfRandomFloats(Size size, T lo = 0, T hi = 1) -> um2::Vector<T>
{
  um2::Vector<T> v(size);
  std::generate(v.begin(), v.end(), randomFloat<T>);
  std::transform(v.begin(), v.end(), v.begin(),
                 [lo, hi](T x) { return lo + x * (hi - lo); });
  return v;
}

template <typename T>
auto
makeVectorOfRandomInts(Size size) -> um2::Vector<T>
{
  um2::Vector<T> v(size);
  std::generate(v.begin(), v.end(), randomInt<T>);
  return v;
}

template <Size D, typename T>
auto
randomPoint() -> um2::Point<D, T>
{
  um2::Point<D, T> p;
  std::generate(p.begin(), p.end(), randomFloat<T>);
  return p;
}

template <Size D, typename T>
auto
makeVectorOfRandomPoints(Size size, um2::AxisAlignedBox<D, T> box)
    -> um2::Vector<um2::Point<D, T>>
{
  um2::Vector<um2::Point<D, T>> v(size);
  std::generate(v.begin(), v.end(), randomPoint<D, T>);
  auto const box_size = box.maxima - box.minima;
  for (auto & p : v) {
    p *= box_size;
    p += box.minima;
  }
  return v;
}

//template <typename T>
//auto
//makeVectorOfRandomTriangles(Size size, um2::AxisAlignedBox2<T> box)
//    -> um2::Vector<um2::Triangle2<T>>
//{
//  um2::Vector<um2::Triangle2<T>> v(size);
//  auto const box_size = box.maxima - box.minima;
//  for (auto & t : v) {
//    t[0] = box.minima + randomPoint<2, T>() *= box_size;
//    t[1] = box.minima + randomPoint<2, T>() *= box_size;
//    // We require that the third point is CCW from the first two.
//    t[2] = box.minima + randomPoint<2, T>() *= box_size;
//    while (!um2::areCCW(t[0], t[1], t[2])) {
//      t[2] = box.minima + randomPoint<2, T>() *= box_size;
//    }
//  }
//  return v;
//}
