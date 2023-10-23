#include <um2/mesh/io_abaqus.hpp>

#include "./helpers/setup_polytope_soup.hpp"

#include "../test_macros.hpp"

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri_mesh)
{
  um2::String const filename = "./mesh_files/tri.inp";
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceTriPolytopeSoup(mesh_ref);

  um2::PolytopeSoup<T, I> mesh;
  um2::readAbaqusFile(filename, mesh);

  ASSERT(um2::compareGeometry(mesh, mesh_ref) == 0);
  ASSERT(um2::compareTopology(mesh, mesh_ref) == 0);
  ASSERT(mesh.elset_names == mesh_ref.elset_names);
  ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets);
  ASSERT(mesh.elset_ids == mesh_ref.elset_ids);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(quad_mesh)
{
  um2::String const filename = "./mesh_files/quad.inp";
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceQuadPolytopeSoup(mesh_ref);

  um2::PolytopeSoup<T, I> mesh;
  um2::readAbaqusFile(filename, mesh);

  ASSERT(um2::compareGeometry(mesh, mesh_ref) == 0);
  ASSERT(um2::compareTopology(mesh, mesh_ref) == 0);
  ASSERT(mesh.elset_names == mesh_ref.elset_names);
  ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets);
  ASSERT(mesh.elset_ids == mesh_ref.elset_ids);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri_quad_mesh)
{
  um2::String const filename = "./mesh_files/tri_quad.inp";
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceTriQuadPolytopeSoup(mesh_ref);

  um2::PolytopeSoup<T, I> mesh;
  um2::readAbaqusFile(filename, mesh);

  ASSERT(um2::compareGeometry(mesh, mesh_ref) == 0);
  ASSERT(um2::compareTopology(mesh, mesh_ref) == 0);
  ASSERT(mesh.elset_names == mesh_ref.elset_names);
  ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets);
  ASSERT(mesh.elset_ids == mesh_ref.elset_ids);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri6_mesh)
{
  um2::String const filename = "./mesh_files/tri6.inp";
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceTri6PolytopeSoup(mesh_ref);

  um2::PolytopeSoup<T, I> mesh;
  um2::readAbaqusFile(filename, mesh);

  ASSERT(um2::compareGeometry(mesh, mesh_ref) == 0);
  ASSERT(um2::compareTopology(mesh, mesh_ref) == 0);
  ASSERT(mesh.elset_names == mesh_ref.elset_names);
  ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets);
  ASSERT(mesh.elset_ids == mesh_ref.elset_ids);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(quad8_mesh)
{
  um2::String const filename = "./mesh_files/quad8.inp";
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceQuad8PolytopeSoup(mesh_ref);

  um2::PolytopeSoup<T, I> mesh;
  um2::readAbaqusFile(filename, mesh);

  ASSERT(um2::compareGeometry(mesh, mesh_ref) == 0);
  ASSERT(um2::compareTopology(mesh, mesh_ref) == 0);
  ASSERT(mesh.elset_names == mesh_ref.elset_names);
  ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets);
  ASSERT(mesh.elset_ids == mesh_ref.elset_ids);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri6_quad8_mesh)
{
  um2::String const filename = "./mesh_files/tri6_quad8.inp";
  um2::PolytopeSoup<T, I> mesh_ref;
  makeReferenceTri6Quad8PolytopeSoup(mesh_ref);

  um2::PolytopeSoup<T, I> mesh;
  um2::readAbaqusFile(filename, mesh);

  ASSERT(um2::compareGeometry(mesh, mesh_ref) == 0);
  ASSERT(um2::compareTopology(mesh, mesh_ref) == 0);
  ASSERT(mesh.elset_names == mesh_ref.elset_names);
  ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets);
  ASSERT(mesh.elset_ids == mesh_ref.elset_ids);
}

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(io_abaqus)
{
  TEST((tri_mesh<T, I>));
  TEST((quad_mesh<T, I>));
  TEST((tri_quad_mesh<T, I>));
  TEST((tri6_mesh<T, I>));
  TEST((quad8_mesh<T, I>));
  TEST((tri6_quad8_mesh<T, I>));
}
auto
main() -> int
{
  RUN_SUITE((io_abaqus<float, int16_t>));
  RUN_SUITE((io_abaqus<float, int32_t>));
  RUN_SUITE((io_abaqus<float, int64_t>));
  RUN_SUITE((io_abaqus<double, int16_t>));
  RUN_SUITE((io_abaqus<double, int32_t>));
  RUN_SUITE((io_abaqus<double, int64_t>));
  return 0;
}
