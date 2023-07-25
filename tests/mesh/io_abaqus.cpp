#include <um2/mesh/io_abaqus.hpp>

#include "./helpers/setup_mesh_file.hpp"

#include "../test_macros.hpp"

#include <algorithm>

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri_mesh)
{
  std::string filename = "./mesh_files/tri.inp";
  um2::MeshFile<T, I> mesh_ref;
  makeReferenceTriMeshFile(mesh_ref);

  um2::MeshFile<T, I> mesh;
  um2::readAbaqusFile(filename, mesh);

  ASSERT(mesh.filepath == mesh_ref.filepath);
  ASSERT(mesh.format == um2::MeshFileFormat::Abaqus);
  ASSERT(mesh.name == mesh_ref.name);
  ASSERT(um2::compareGeometry(mesh, mesh_ref) == 0);
  ASSERT(mesh.element_types == mesh_ref.element_types);
  ASSERT(mesh.elset_names == mesh_ref.elset_names);
  ASSERT(mesh.elset_ids == mesh_ref.elset_ids);
}

// template <std::floating_point T, std::signed_integral I>
// TEST(quad_mesh)
//     std::string filename = "./test/mesh/mesh_files/quad.inp";
//     um2::MeshFile<T, I> mesh_ref;
//     make_quad_reference_mesh_file(mesh_ref);
//     um2::MeshFile<T, I> mesh;
//     um2::read_abaqus_file(filename, mesh);
//     ASSERT(mesh.filepath == mesh_ref.filepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::ABAQUS, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     ASSERT(mesh.element_types == mesh_ref.element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(tri_quad_mesh)
//     std::string filename = "./test/mesh/mesh_files/tri_quad.inp";
//     um2::MeshFile<T, I> mesh_ref;
//     make_tri_quad_reference_mesh_file(mesh_ref);
//     um2::MeshFile<T, I> mesh;
//     um2::read_abaqus_file(filename, mesh);
//     ASSERT(mesh.filepath == mesh_ref.filepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::ABAQUS, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     ASSERT(mesh.element_types == mesh_ref.element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(tri6_mesh)
//     std::string filename = "./test/mesh/mesh_files/tri6.inp";
//     um2::MeshFile<T, I> mesh_ref;
//     make_tri6_reference_mesh_file(mesh_ref);
//     um2::MeshFile<T, I> mesh;
//     um2::read_abaqus_file(filename, mesh);
//     ASSERT(mesh.filepath == mesh_ref.filepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::ABAQUS, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     ASSERT(mesh.element_types == mesh_ref.element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(quad8_mesh)
//     std::string filename = "./test/mesh/mesh_files/quad8.inp";
//     um2::MeshFile<T, I> mesh_ref;
//     make_quad8_reference_mesh_file(mesh_ref);
//     um2::MeshFile<T, I> mesh;
//     um2::read_abaqus_file(filename, mesh);
//     ASSERT(mesh.filepath == mesh_ref.filepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::ABAQUS, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     ASSERT(mesh.element_types == mesh_ref.element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(tri6_quad8_mesh)
//     std::string filename = "./test/mesh/mesh_files/tri6_quad8.inp";
//     um2::MeshFile<T, I> mesh_ref;
//     make_tri6_quad8_reference_mesh_file(mesh_ref);
//     um2::MeshFile<T, I> mesh;
//     um2::read_abaqus_file(filename, mesh);
//     ASSERT(mesh.filepath == mesh_ref.filepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::ABAQUS, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     ASSERT(mesh.element_types == mesh_ref.element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(io_abaqus)
{
  TEST((tri_mesh<T, I>));
  //    RUN_TEST("quad_mesh", (quad_mesh<T, I>) );
  //    RUN_TEST("tri_quad_mesh", (tri_quad_mesh<T, I>) );
  //    RUN_TEST("tri6_mesh", (tri6_mesh<T, I>) );
  //    RUN_TEST("quad8_mesh", (quad8_mesh<T, I>) );
  //    RUN_TEST("tri6_quad8_mesh", (tri6_quad8_mesh<T, I>) );
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
