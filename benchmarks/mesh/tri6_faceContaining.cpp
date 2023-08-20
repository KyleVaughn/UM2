//=============================================================================
// Findings
//=============================================================================
//
//faceContaining268/16384               19.8 ms         19.8 ms           36
//faceContaining268/65536               77.1 ms         77.1 ms            9
//binnedFaceContaining268/16384         2.84 ms         2.84 ms          246
//binnedFaceContaining268/65536         11.3 ms         11.3 ms           62
//faceContaining1656/16384               188 ms          188 ms            4
//faceContaining1656/65536               763 ms          763 ms            1
//binnedFaceContaining1656/16384        3.99 ms         3.99 ms          175
//binnedFaceContaining1656/65536        15.9 ms         15.9 ms           44
//faceContaining76790/16384             9212 ms         9212 ms            1
//faceContaining76790/65536            36515 ms        36512 ms            1
//binnedFaceContaining76790/16384       7.12 ms         7.12 ms           97
//binnedFaceContaining76790/65536       28.5 ms         28.5 ms           25
//
// Obviously, as the mesh gets larger, the binned version becomes much faster, since
// the regular version is O(N) and the binned version is O(1)
// Speedup for   268:    7x
// Speedup for  1656:   47x
// Speedup for 76790: 1280x
// The time per point for the binned version in on the order of hundreds of nanoseconds while
// the regular version is on the order of 1 to hundreds of microseconds.

#include <um2/mesh/BinnedFaceVertexMesh.hpp>
#include <um2/mesh/QuadraticTriMesh.hpp>
#include <um2/mesh/io.hpp>

#include "../helpers.hpp"

#include <iostream>

constexpr Size npoints = 1048576 / 16;

class Pin268Fixture : public benchmark::Fixture
{
public:
  um2::QuadraticTriMesh<2, float, int32_t> mesh;

  void
  SetUp(const ::benchmark::State & /*state*/) final
  {
    um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
    std::string const filename = "./mesh_files/tri6_pin_268.inp";
    um2::MeshFile<float, int32_t> meshfile;
    um2::readAbaqusFile(filename, meshfile);
    mesh = um2::QuadraticTriMesh<2, float, int32_t>(meshfile);
  }

  void
  TearDown(const ::benchmark::State & /*state*/) final
  {
  }
};

class Pin1656Fixture : public benchmark::Fixture
{
public:
  um2::QuadraticTriMesh<2, float, int32_t> mesh;

  void
  SetUp(const ::benchmark::State & /*state*/) final
  {
    um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
    std::string const filename = "./mesh_files/tri6_pin_1656.inp";
    um2::MeshFile<float, int32_t> meshfile;
    um2::readAbaqusFile(filename, meshfile);
    mesh = um2::QuadraticTriMesh<2, float, int32_t>(meshfile);
  }

  void
  TearDown(const ::benchmark::State & /*state*/) final
  {
  }
};

class Lattice76790Fixture : public benchmark::Fixture
{
 public:
   um2::QuadraticTriMesh<2, float, int32_t> mesh;

   void
   SetUp(const ::benchmark::State & /*state*/) final
   {
     um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
     std::string const filename = "./mesh_files/tri6_lattice_76790.inp";
     um2::MeshFile<float, int32_t> meshfile;
     um2::readAbaqusFile(filename, meshfile);
     mesh = um2::QuadraticTriMesh<2, float, int32_t>(meshfile);
   }

   void
   TearDown(const ::benchmark::State & /*state*/) final
   {
   }
 };

BENCHMARK_DEFINE_F(Pin268Fixture, faceContaining268)(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<float> const box = mesh.boundingBox();
  auto const points = makeVectorOfRandomPoints(n, box);
  // NOLINTNEXTLINE
  for (auto s : state) {
    for (auto const & p : points) {
      Size const i = mesh.faceContaining(p);
      if (i == -1) {
        std::cerr << "Face not found" << std::endl;
      }
    }
  }
}

BENCHMARK_DEFINE_F(Pin268Fixture, binnedFaceContaining268)(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<float> const box = mesh.boundingBox();
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::BinnedQuadraticTriMesh<2, float, int32_t> const binnedmesh(mesh);
  // NOLINTNEXTLINE
  for (auto s : state) {
    for (auto const & p : points) {
      Size const i = um2::faceContaining(binnedmesh, p);
      if (i == -1) {
        std::cerr << "Face not found" << std::endl;
      }
    }
  }
}

BENCHMARK_DEFINE_F(Pin1656Fixture, faceContaining1656)(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<float> const box = mesh.boundingBox();
  auto const points = makeVectorOfRandomPoints(n, box);
  // NOLINTNEXTLINE
  for (auto s : state) {
    for (auto const & p : points) {
      Size const i = mesh.faceContaining(p);
      if (i == -1) {
        std::cerr << "Face not found" << std::endl;
      }
    }
  }
}

BENCHMARK_DEFINE_F(Pin1656Fixture, binnedFaceContaining1656)(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<float> const box = mesh.boundingBox();
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::BinnedQuadraticTriMesh<2, float, int32_t> const binnedmesh(mesh);
  // NOLINTNEXTLINE
  for (auto s : state) {
    for (auto const & p : points) {
      Size const i = um2::faceContaining(binnedmesh, p);
      if (i == -1) {
        std::cerr << "Face not found" << std::endl;
      }
    }
  }
}

BENCHMARK_DEFINE_F(Lattice76790Fixture, faceContaining76790)(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<float> const box = mesh.boundingBox();
  auto const points = makeVectorOfRandomPoints(n, box);
  // NOLINTNEXTLINE
  for (auto s : state) {
    for (auto const & p : points) {
      Size const i = mesh.faceContaining(p);
      if (i == -1) {
        std::cerr << "Face not found" << std::endl;
      }
    }
  }
}

BENCHMARK_DEFINE_F(Lattice76790Fixture, binnedFaceContaining76790)(benchmark::State & state)
{
  Size const n = static_cast<Size>(state.range(0));
  um2::AxisAlignedBox2<float> const box = mesh.boundingBox();
  auto const points = makeVectorOfRandomPoints(n, box);
  um2::BinnedQuadraticTriMesh<2, float, int32_t> const binnedmesh(mesh);
  // NOLINTNEXTLINE
  for (auto s : state) {
    for (auto const & p : points) {
      Size const i = um2::faceContaining(binnedmesh, p); 
      if (i == -1) {
        std::cerr << "Face not found" << std::endl;
      }
    }
  }
}

BENCHMARK_REGISTER_F(Pin268Fixture, faceContaining268)
   ->RangeMultiplier(4)
   ->Range(16384, npoints)
   ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Pin268Fixture, binnedFaceContaining268)
    ->RangeMultiplier(4)
    ->Range(16384, npoints)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Pin1656Fixture, faceContaining1656)
   ->RangeMultiplier(4)
   ->Range(16384, npoints)
   ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Pin1656Fixture, binnedFaceContaining1656)
   ->RangeMultiplier(4)
   ->Range(16384, npoints)
   ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Lattice76790Fixture, faceContaining76790)
   ->RangeMultiplier(4)
   ->Range(16384, npoints)
   ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(Lattice76790Fixture, binnedFaceContaining76790)
  ->RangeMultiplier(4)
  ->Range(16384, npoints)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
