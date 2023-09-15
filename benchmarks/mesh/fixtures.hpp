#include <benchmark/benchmark.h>
#include <um2/mesh/BinnedFaceVertexMesh.hpp>
#include <um2/mesh/FaceVertexMesh.hpp>
#include <um2/mesh/io.hpp>

class Pin268Fixture : public benchmark::Fixture
{
public:
  um2::MeshFile<float, int32_t> meshfile;
  um2::QuadraticTriMesh<2, float, int32_t> mesh;

  void
  SetUp(const ::benchmark::State & /*state*/) final
  {
    um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
    std::string const filename = "./mesh_files/tri6_pin_268.inp";
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
  um2::MeshFile<float, int32_t> meshfile;
  um2::QuadraticTriMesh<2, float, int32_t> mesh;

  void
  SetUp(const ::benchmark::State & /*state*/) final
  {
    um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
    std::string const filename = "./mesh_files/tri6_pin_1656.inp";
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
  um2::MeshFile<float, int32_t> meshfile;
  um2::QuadraticTriMesh<2, float, int32_t> mesh;

  void
  SetUp(const ::benchmark::State & /*state*/) final
  {
    um2::Log::setMaxVerbosityLevel(um2::LogVerbosity::Warn);
    std::string const filename = "./mesh_files/tri6_lattice_76790.inp";
    um2::readAbaqusFile(filename, meshfile);
    mesh = um2::QuadraticTriMesh<2, float, int32_t>(meshfile);
  }

  void
  TearDown(const ::benchmark::State & /*state*/) final
  {
  }
};
