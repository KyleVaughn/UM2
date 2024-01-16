#include <benchmark/benchmark.h>
#include <um2/mesh/face_vertex_mesh.hpp>

template <std::floating_point T, std::signed_integral I>
class Tri6Fixture : public benchmark::Fixture
{
public:
  um2::PolytopeSoup<T, I> small_pin_soup;
  um2::PolytopeSoup<T, I> big_pin_soup;
  um2::PolytopeSoup<T, I> small_lattice_soup;
  um2::PolytopeSoup<T, I> big_lattice_soup;
  um2::QuadraticTriMesh<2, T, I> small_pin;
  um2::QuadraticTriMesh<2, T, I> big_pin;
  um2::QuadraticTriMesh<2, T, I> small_lattice;
  um2::QuadraticTriMesh<2, T, I> big_lattice;

  void
  SetUp(const ::benchmark::State & /*state*/) final
  {
    if (small_pin.numVertices() == 0) {
      small_pin_soup.read("./mesh_files/tri6_pin_268.inp");
      big_pin_soup.read("./mesh_files/tri6_pin_1656.inp");
      small_lattice_soup.read("./mesh_files/tri6_lattice_76790.inp");
      big_lattice_soup.read("./mesh_files/tri6_lattice_478194.inp");
      //      small_pin_soup.mortonSort();
      //      big_pin_soup.mortonSort();
      //      small_lattice_soup.mortonSort();
      //      big_lattice_soup.mortonSort();
      small_pin = um2::QuadraticTriMesh<2, T, I>(small_pin_soup);
      big_pin = um2::QuadraticTriMesh<2, T, I>(big_pin_soup);
      small_lattice = um2::QuadraticTriMesh<2, T, I>(small_lattice_soup);
      big_lattice = um2::QuadraticTriMesh<2, T, I>(big_lattice_soup);
    }
  }

  void
  TearDown(const ::benchmark::State & /*state*/) final
  {
  }
};
