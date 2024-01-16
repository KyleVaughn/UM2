//=============================================================================
// Summary
//=============================================================================
//
// Purpose:
// -------
// This benchmark tests the performance of various algorithms for computing
// intersections between rays and a quadratic triangle mesh.
//
// Description:
// ------------
// Test the performance of:
// - Computing intersections by materializing each face, then intersecting each
//   edge with the ray.
// - The effect of sorting the vertices and faces by Morton code.
// - Computing intersections by using the connectivity of each face to materialize
//   the edges only.
// - 32-bit vs 64-bit floats and integers.
// - CPU vs GPU.
// by shooting modular rays at a quadratic triangle mesh of various sizes.
//
// Results:
// --------
// float32, int32, face iteration, no sorting, CPU:
// |  Faces  |  Rays  |  Time (ms) |  Rays/s  | Intersections  | Intersections/s  |
// +---------+--------+------------+----------+----------------+------------------+
// |     268 |    664 |       5.1  |  130,196 |         25,786 |        5,056,078 |
// |   1,656 |    664 |      45.3  |   14,657 |         60,745 |        1,340,949 |
// |  76,790 | 11,196 |   28119.   |      398 |      7,354,547 |          261,569 |
// | 478,194 | 11,196 |  207415.   |       54 |     17,391,061 |           83,855 |
// +---------+--------+------------+----------+----------------+------------------+
//
// Modified the method to use fixed-size buffers for the intersections instead of
// push_back. This gave about 20% - 30% improvement, so we use this method from now on.
// (4.27 ms, 37.8 ms, 22684 ms, didn't run)
//
// Materialized each edge instead of each face. Some cases were slower, some faster.
// (4.29 ms, 36.1 ms, 23183 ms, didn't run)
//
// Morton sorted. The small cases are faster (5.4%, 11.5%), but he large case is slower
// (-11.9%). (3.89 ms, 33.9 ms, 26295 ms, didn't run)
//
// float64, int32, face iteration, morton sorted, CPU:
// (3.98 ms, 32.2 ms, 26233 ms, didn't run)
//
// float64, int64, face iteration, morton sorted, CPU:
// (4.00 ms, 31.3 ms, 26710 ms, didn't run)
//
// Conclusions:
// ------------
// - float and int size don't matter much.
// - Sorting helps for small meshes, but may hurt for large meshes.
// - Materializing faces vs edges doesn't matter much.
//
// TODO(kcvaughn): Add GPU tests.

#include "../helpers.hpp"
#include "fixtures.hpp"
#include <iostream>

constexpr Size nangles = 4;       // equally spaced angles in (0, pi)
constexpr Size rays_per_cm = 100; // number of rays per cm

BENCHMARK_TEMPLATE_DEFINE_F(Tri6Fixture, smallPin_f32i32, float, int32_t)
(benchmark::State & state)
{
  using T = float;
  um2::AxisAlignedBox2<T> const box = small_pin.boundingBox();
  T const s = static_cast<T>(1) / static_cast<T>(rays_per_cm);
  Size constexpr buffer_size = 256;
  Size n = buffer_size;
  um2::Vector<T> intersections(buffer_size);
  bool first_print = true;
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (auto _ : state) {
    Size total_intersections = 0;
    Size total_rays = 0;
    // For each angle
    for (Size ia = 0; ia < nangles; ++ia) {
      T const a = (um2::pi_2<T> / nangles) * static_cast<T>(2 * ia + 1);
      um2::ModularRayParams<T> const params(a, s, box);
      Size const nrays = params.getTotalNumRays();
      total_rays += nrays;
      // For each ray
      for (Size ir = 0; ir < nrays; ++ir) {
        auto const ray = params.getRay(ir);
        //        small_pin.intersect(ray, intersections);
        //        total_intersections += intersections.size();
        //        intersections.resize(0);
        um2::intersectFixedBuffer(ray, small_pin, intersections.data(), &n);
        total_intersections += n;
        n = buffer_size;
      } // end for each ray
    }   // end for each angle
    if (first_print) {
      std::cout << "Total rays and intersections: " << total_rays << ", "
                << total_intersections << std::endl;
      first_print = false;
    }
  } // end for each state
}

BENCHMARK_TEMPLATE_DEFINE_F(Tri6Fixture, bigPin_f32i32, float, int32_t)
(benchmark::State & state)
{
  using T = float;
  um2::AxisAlignedBox2<T> const box = big_pin.boundingBox();
  T const s = static_cast<T>(1) / static_cast<T>(rays_per_cm);
  Size constexpr buffer_size = 1024;
  Size n = buffer_size;
  um2::Vector<T> intersections(buffer_size);
  bool first_print = true;
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (auto _ : state) {
    Size total_intersections = 0;
    Size total_rays = 0;
    // For each angle
    for (Size ia = 0; ia < nangles; ++ia) {
      T const a = (um2::pi_2<T> / nangles) * static_cast<T>(2 * ia + 1);
      um2::ModularRayParams<T> const params(a, s, box);
      Size const nrays = params.getTotalNumRays();
      total_rays += nrays;
      // For each ray
      for (Size ir = 0; ir < nrays; ++ir) {
        auto const ray = params.getRay(ir);
        //        big_pin.intersect(ray, intersections);
        //        total_intersections += intersections.size();
        //        intersections.resize(0);
        um2::intersectFixedBuffer(ray, big_pin, intersections.data(), &n);
        total_intersections += n;
        n = buffer_size;
      } // end for each ray
    }   // end for each angle
    if (first_print) {
      std::cout << "Total rays and intersections: " << total_rays << ", "
                << total_intersections << std::endl;
      first_print = false;
    }
  } // end for each state
}

BENCHMARK_TEMPLATE_DEFINE_F(Tri6Fixture, smallLattice_f32i32, float, int32_t)
(benchmark::State & state)
{
  using T = float;
  um2::AxisAlignedBox2<T> const box = small_lattice.boundingBox();
  T const s = static_cast<T>(1) / static_cast<T>(rays_per_cm);
  Size constexpr buffer_size = 2048;
  Size n = buffer_size;
  um2::Vector<T> intersections(buffer_size);
  bool first_print = true;
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (auto _ : state) {
    Size total_intersections = 0;
    Size total_rays = 0;
    // For each angle
    for (Size ia = 0; ia < nangles; ++ia) {
      T const a = (um2::pi_2<T> / nangles) * static_cast<T>(2 * ia + 1);
      um2::ModularRayParams<T> const params(a, s, box);
      Size const nrays = params.getTotalNumRays();
      total_rays += nrays;
      // For each ray
      for (Size ir = 0; ir < nrays; ++ir) {
        auto const ray = params.getRay(ir);
        //        small_lattice.intersect(ray, intersections);
        //        total_intersections += intersections.size();
        //        intersections.resize(0);
        um2::intersectFixedBuffer(ray, small_lattice, intersections.data(), &n);
        total_intersections += n;
        n = buffer_size;
      } // end for each ray
    }   // end for each angle
    if (first_print) {
      std::cout << "Total rays and intersections: " << total_rays << ", "
                << total_intersections << std::endl;
      first_print = false;
    }
  } // end for each state
}

BENCHMARK_TEMPLATE_DEFINE_F(Tri6Fixture, bigLattice_f32i32, float, int32_t)
(benchmark::State & state)
{
  using T = float;
  um2::AxisAlignedBox2<T> const box = big_lattice.boundingBox();
  T const s = static_cast<T>(1) / static_cast<T>(rays_per_cm);
  Size constexpr buffer_size = 4096;
  Size n = buffer_size;
  um2::Vector<T> intersections(buffer_size);
  bool first_print = true;
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  for (auto _ : state) {
    Size total_intersections = 0;
    Size total_rays = 0;
    // For each angle
    for (Size ia = 0; ia < nangles; ++ia) {
      T const a = (um2::pi_2<T> / nangles) * static_cast<T>(2 * ia + 1);
      um2::ModularRayParams<T> const params(a, s, box);
      Size const nrays = params.getTotalNumRays();
      total_rays += nrays;
      // For each ray
      for (Size ir = 0; ir < nrays; ++ir) {
        auto const ray = params.getRay(ir);
        //        big_lattice.intersect(ray, intersections);
        //        total_intersections += intersections.size();
        //        intersections.resize(0);
        um2::intersectFixedBuffer(ray, big_lattice, intersections.data(), &n);
        total_intersections += n;
        n = buffer_size;
      } // end for each ray
    }   // end for each angle
    if (first_print) {
      std::cout << "Total rays and intersections: " << total_rays << ", "
                << total_intersections << std::endl;
      first_print = false;
    }
  } // end for each state
}

BENCHMARK_REGISTER_F(Tri6Fixture, smallPin_f32i32)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Tri6Fixture, bigPin_f32i32)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Tri6Fixture, smallLattice_f32i32)->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Tri6Fixture, bigLattice_f32i32)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
