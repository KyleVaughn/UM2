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
  void * d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cudaError_t error = cudaSuccess;
  error = cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, begin, num_points,
                                 MortonLessFunctor<U, D, T>{});
  CUDA_CHECK_ERROR(error);

  // Allocate temporary storage
  error = cudaMalloc(&d_temp_storage, temp_storage_bytes);
  CUDA_CHECK_ERROR(error);

  // Sort the keys
  error = cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, begin, num_points,
                                 MortonLessFunctor<U, D, T>{});
  CUDA_CHECK_ERROR(error);

  // Free temporary storage
  error = cudaFree(d_temp_storage);
  CUDA_CHECK_ERROR(error);
}
#endif

} // namespace um2::parallel
