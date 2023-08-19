//// FINDINGS:
//// Using the AABB method for isLeft
//// Pin268FaceFixture/faceContaining268/16384/min_time:10.000           28.8
//// ms         28.8 ms          491
//// Pin268FaceFixture/faceContaining268/65536/min_time:10.000            115 ms 115 ms
/// 121 / Pin268FaceFixture/faceContaining268/262144/min_time:10.000           462 ms 462
/// ms 30 / Pin268FaceFixture/faceContaining268/1048576/min_time:10.000         1836 ms
/// 1835 ms 8 / Pin1656FaceFixture/faceContaining1656/16384/min_time:10.000          260
/// ms 260 ms 54 / Using the Bezier bounding Triangle method for isLeft /
/// Pin268FaceFixture/faceContaining268/16384/min_time:10.000           13.5 /
/// ms         13.5 ms         1044 /
/// Pin268FaceFixture/faceContaining268/65536/min_time:10.000           54.2 /
/// ms         54.2 ms          259 /
/// Pin268FaceFixture/faceContaining268/262144/min_time:10.000           217 ms 217 ms 64
//// Pin268FaceFixture/faceContaining268/1048576/min_time:10.000          865 ms 865 ms 16
//// Pin1656FaceFixture/faceContaining1656/16384/min_time:10.000          148 ms 148 ms 95
//
//// Looks like the Bezier bounding triangle method is faster

#include "../helpers.hpp"
#include <um2/mesh/QuadraticTriMesh.hpp>
#include <um2/mesh/io.hpp>

#include <iostream>

// constexpr Size npoints = 1048576;
//
// class Pin268FaceFixture : public benchmark::Fixture
//{
// public:
//   um2::QuadraticTriMesh<2, float, int32_t> mesh;
//
//   void
//   SetUp(const ::benchmark::State & /*state*/) final
//   {
//     um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
//     std::string const filename = "./mesh_files/tri6_pin_268.inp";
//     um2::MeshFile<float, int32_t> meshfile;
//     um2::readAbaqusFile(filename, meshfile);
//     mesh = um2::QuadraticTriMesh<2, float, int32_t>(meshfile);
//   }
//
//   void
//   TearDown(const ::benchmark::State & /*state*/) final
//   {
//   }
// };
//
// class Pin1656FaceFixture : public benchmark::Fixture
//{
// public:
//   um2::QuadraticTriMesh<2, float, int32_t> mesh;
//
//   void
//   SetUp(const ::benchmark::State & /*state*/) final
//   {
//     um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
//     std::string const filename = "./mesh_files/tri6_pin_1656.inp";
//     um2::MeshFile<float, int32_t> meshfile;
//     um2::readAbaqusFile(filename, meshfile);
//     mesh = um2::QuadraticTriMesh<2, float, int32_t>(meshfile);
//   }
//
//   void
//   TearDown(const ::benchmark::State & /*state*/) final
//   {
//   }
// };
//
//// class LatticeFaceFixture : public benchmark::Fixture
////{
//// public:
////   um2::QuadraticTriMesh<2, float, int32_t> mesh;
////
////   void
////   SetUp(const ::benchmark::State & /*state*/) final
////   {
////     um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
////     std::string const filename = "./mesh_files/tri6_pin_1656.inp";
////     um2::MeshFile<float, int32_t> meshfile;
////     um2::readAbaqusFile(filename, meshfile);
////     mesh = um2::QuadraticTriMesh<2, float, int32_t>(meshfile);
////   }
////
////   void
////   TearDown(const ::benchmark::State & /*state*/) final
////   {
////   }
//// };
//
//// cppcheck-suppress unknownMacro
// BENCHMARK_DEFINE_F(Pin268FaceFixture, faceContaining268)(benchmark::State & state)
//{
//   Size const n = static_cast<Size>(state.range(0));
//   um2::Vector<um2::Point2<float>> const points =
//       makeVectorOfRandomPoints<2, float, 0, 1>(n);
//   // NOLINTNEXTLINE
//   for (auto s : state) {
//     for (auto const & p : points) {
//       Size const i = mesh.faceContaining(p);
//       if (i == -1) {
//         std::cerr << "Face not found" << std::endl;
//       }
//     }
//   }
// }
//
// BENCHMARK_DEFINE_F(Pin1656FaceFixture, faceContaining1656)(benchmark::State & state)
//{
//   Size const n = static_cast<Size>(state.range(0));
//   um2::Vector<um2::Point2<float>> const points =
//       makeVectorOfRandomPoints<2, float, 0, 1>(n);
//   // NOLINTNEXTLINE
//   for (auto s : state) {
//     for (auto const & p : points) {
//       Size const i = mesh.faceContaining(p);
//       if (i == -1) {
//         std::cerr << "Face not found" << std::endl;
//       }
//     }
//   }
// }
//
//// #if _OPENMP
//// template <typename T, typename U>
//// static void
//// mortonSortParallel(benchmark::State & state)
////{
////   Size const n = static_cast<Size>(state.range(0));
////   um2::Vector<um2::Point<dim, T>> points = makeVectorOfRandomPoints<dim, T, 0, 1>(n);
////   std::random_device rd;
////   std::mt19937 g(rd());
////   // NOLINTNEXTLINE
////   for (auto s : state) {
////     state.PauseTiming();
////     std::shuffle(points.begin(), points.end(), g);
////     state.ResumeTiming();
////     __gnu_parallel::sort(points.begin(), points.end(), um2::mortonLess<U, dim, T>);
////   }
////   if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, dim, T>)) {
////     std::cout << "Not sorted" << std::endl;
////   }
//// }
////
////// template <typename T, typename U>
////// static void mortonSortParallelExec(benchmark::State& state)
//////{
//////   Size const n = static_cast<Size>(state.range(0));
//////   um2::Vector<um2::Point<dim, T>> points = makeVectorOfRandomPoints<dim, T, 0,
/// 1>(n);
//////   // NOLINTNEXTLINE
//////   for (auto s : state) {
//////     state.PauseTiming();
//////     std::random_shuffle(points.begin(), points.end());
//////     state.ResumeTiming();
//////     std::sort(std::execution::par_unseq, points.begin(), points.end(),
//////     um2::mortonLess<U, dim, T>);
//////   }
//////   if (!std::is_sorted(points.begin(), points.end(), um2::mortonLess<U, dim, T>)) {
//////     std::cout << "Not sorted" << std::endl;
//////   }
////// }
//// #endif
////
//// #if UM2_ENABLE_CUDA
//// template <typename T, typename U>
//// static void
//// mortonSortCuda(benchmark::State & state)
////{
////   Size const n = static_cast<Size>(state.range(0));
////   um2::Vector<um2::Point<dim, T>> points = makeVectorOfRandomPoints<dim, T, 0, 1>(n);
////   um2::Vector<um2::Point<dim, T>> after(n);
////
////   um2::Point2<T> * d_points;
////   transferToDevice(&d_points, points);
////
////   // NOLINTNEXTLINE
////   for (auto s : state) {
////     state.PauseTiming();
////     // we don't have a random_shuffle for CUDA, so just copy the points
////     // to the device again
////     transferToDevice(&d_points, points);
////     state.ResumeTiming();
////     um2::deviceMortonSort<U>(d_points, d_points + points.size());
////     cudaDeviceSynchronize();
////   }
////
////   transferFromDevice(after, d_points);
////   if (!std::is_sorted(after.begin(), after.end(), um2::mortonLess<U, dim, T>)) {
////     std::cout << "Not sorted" << std::endl;
////   }
////   cudaFree(d_points);
//// }
//// #endif
//
// BENCHMARK_REGISTER_F(Pin268FaceFixture, faceContaining268)
//    ->RangeMultiplier(4)
//    ->Range(16384, npoints)
//    ->Unit(benchmark::kMillisecond)
//    ->MinTime(10);
//
// BENCHMARK_REGISTER_F(Pin1656FaceFixture, faceContaining1656)
//    ->RangeMultiplier(4)
//    ->Range(16384, npoints)
//    ->Unit(benchmark::kMillisecond)
//    ->MinTime(10);

// #if _OPENMP
// BENCHMARK_TEMPLATE2(mortonSortParallel, float, uint32_t)
//     ->RangeMultiplier(4)
//     ->Range(16, npoints)
//     ->Unit(benchmark::kMicrosecond);
//// BENCHMARK_TEMPLATE2(mortonSortParallelExec, float, uint32_t)
////   ->RangeMultiplier(4)
////   ->Range(16, npoints)
////   ->Unit(benchmark::kMicrosecond);
// #endif
// #if UM2_ENABLE_CUDA
// BENCHMARK_TEMPLATE2(mortonSortCuda, float, uint32_t)
//     ->RangeMultiplier(4)
//     ->Range(16, npoints)
//     ->Unit(benchmark::kMicrosecond);
// #endif
BENCHMARK_MAIN();
