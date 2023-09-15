#pragma once

#include <um2/geometry/morton_sort_points.hpp>

#if UM2_USE_TBB
#  include <execution>
#endif

#if UM2_USE_CUDA
#  include <cub/device/device_merge_sort.cuh>
#endif

namespace um2::parallel
{

#if UM2_USE_TBB
template <std::unsigned_integral U, Size D, std::floating_point T>
void
mortonSort(Point<D, T> * const begin, Point<D, T> * const end)
{
  std::sort(std::execution::par_unseq, begin, end, mortonLess<U, D, T>);
}
#endif

#if UM2_USE_CUDA
template <std::unsigned_integral U, Size D, std::floating_point T>
void
deviceMortonSort(Point<D, T> * const begin, Point<D, T> * const end)
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

} // namespace um2::parallel
