#pragma once

#include <um2/geometry/Point.hpp>
#include <um2/math/morton.hpp>

#include <algorithm> // std::sort

#ifdef _OPENMP
#  include <parallel/algorithm> // __gnu_parallel::sort
#endif

#if UM2_ENABLE_CUDA
#  include <cub/device/device_merge_sort.cuh>
#endif

namespace um2
{

// -----------------------------------------------------------------------------
// Morton encoding/decoding with normalization
// -----------------------------------------------------------------------------
template <std::unsigned_integral U, Size D, std::floating_point T>
PURE HOSTDEV auto
mortonEncode(Point<D, T> const & p) -> U
{
  if constexpr (D == 2) {
    return mortonEncode<U>(p[0], p[1]);
  } else if constexpr (D == 3) {
    return mortonEncode<U>(p[0], p[1], p[2]);
  } else {
    static_assert(D == 2 || D == 3);
    return 0;
  }
}

template <std::unsigned_integral U, Size D, std::floating_point T>
HOSTDEV void
mortonDecode(U const morton, Point<D, T> & p)
{
  if constexpr (D == 2) {
    mortonDecode(morton, p[0], p[1]);
  } else if constexpr (D == 3) {
    mortonDecode(morton, p[0], p[1], p[2]);
  } else {
    static_assert(D == 2 || D == 3);
  }
}

template <std::unsigned_integral U, Size D, std::floating_point T>
PURE HOSTDEV auto
mortonLess(Point<D, T> const & lhs, Point<D, T> const & rhs) -> bool
{
  return mortonEncode<U>(lhs) < mortonEncode<U>(rhs);
}

template <std::unsigned_integral U, Size D, std::floating_point T>
struct MortonLessFunctor {
  PURE HOSTDEV auto
  operator()(Point<D, T> const & lhs, Point<D, T> const & rhs) const -> bool
  {
    return mortonEncode<U>(lhs) < mortonEncode<U>(rhs);
  }
};

template <std::unsigned_integral U, Size D, std::floating_point T>
void
mortonSort(Point<D, T> * begin, Point<D, T> * end)
{
#ifdef _OPENMP
  __gnu_parallel::sort(begin, end, mortonLess<U, D, T>);
#else
  std::sort(begin, end, mortonLess<U, D, T>);
#endif
}

#if UM2_ENABLE_CUDA
template <std::unsigned_integral U, Size D, std::floating_point T>
void
deviceMortonSort(Point<D, T> * begin, Point<D, T> * end)
{
  // Compute the number of elements to sort
  auto const num_points = static_cast<size_t>(end - begin);

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceMergeSort::SortKeys(nullptr, temp_storage_bytes, begin, num_points,
                                 MortonLessFunctor<U, D, T>{});

  // Allocate temporary storage
  void * d_temp_storage = nullptr;
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Sort the keys
  cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, begin, num_points,
                                 MortonLessFunctor<U, D, T>{});

  // Free temporary storage
  cudaFree(d_temp_storage);
}
#endif

} // namespace um2
