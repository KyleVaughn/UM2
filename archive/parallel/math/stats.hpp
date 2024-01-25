#pragma once

#include <um2/stdlib/algorithm.hpp>
#include <um2/stdlib/math.hpp>

#if UM2_USE_TBB
#  include <execution>
#endif

namespace um2::parallel
{

//=============================================================================
// mean
//=============================================================================

template <std::floating_point T>
constexpr auto
mean(T const * begin, T const * end) -> T
{
  assert(begin != end);
  return std::reduce(std::execution::par_unseq, begin, end, static_cast<T>(0)) /
         static_cast<T>(end - begin);
}

//=============================================================================
// variance
//=============================================================================

template <std::floating_point T>
constexpr auto
variance(T const * begin, T const * end) -> T
{
  assert(begin != end);
  auto const m = um2::parallel::mean(begin, end);
  return std::transform_reduce(std::execution::par_unseq, begin, end, static_cast<T>(0),
                               std::plus<T>{},
                               [m](auto const x) { return (x - m) * (x - m); }) /
         static_cast<T>(end - begin - 1);
}

//=============================================================================
// stdDev
//=============================================================================

template <std::floating_point T>
constexpr auto
stdDev(T const * begin, T const * end) -> T
{
  return um2::sqrt(um2::parallel::variance(begin, end));
}

} // namespace um2::parallel
