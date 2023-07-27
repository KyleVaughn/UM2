#include <um2/mesh/io_xdmf.hpp>

#include "./helpers/setup_mesh_file.hpp"

#include "../test_macros.hpp"

#include <iostream>

template <std::floating_point T, std::signed_integral I>
TEST_CASE(tri_mesh)
{
  um2::MeshFile<T, I> mesh_ref;
  makeReferenceTriMeshFile(mesh_ref);
  mesh_ref.filepath = "./tri.inp";
  um2::writeXDMFFile<T, I>(mesh_ref);

  um2::MeshFile<T, I> mesh;
  um2::readXDMFFile("./tri.xdmf", mesh);
  ASSERT(mesh.filepath == "./tri.xdmf");
  ASSERT(mesh.format == um2::MeshFileFormat::XDMF);
  ASSERT(mesh.name == mesh_ref.name);
  ASSERT(um2::compareGeometry(mesh, mesh_ref) == 0);
  ASSERT(um2::compareTopology(mesh, mesh_ref) == 0);
  std::vector<int8_t> const element_types = {4, 4};
  ASSERT(mesh.element_types == element_types);
  ASSERT(mesh.elset_names == mesh_ref.elset_names);
  ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets);
  ASSERT(mesh.elset_ids == mesh_ref.elset_ids);
}

// template <std::floating_point T, std::signed_integral I>
// TEST(quad_mesh)
//     um2::MeshFile<T, I> mesh_ref;
//     make_quad_reference_mesh_file(mesh_ref);
//     mesh_ref.filepath = std::filesystem::current_path().string() +
//     "/xdmf_test/quad.inp"; um2::write_xdmf_file<T, I>(mesh_ref); std::string xfilepath
//     =
//         std::filesystem::current_path().string() + "/xdmf_test/quad.xdmf";
//     um2::MeshFile<T, I> mesh;
//     um2::read_xdmf_file(xfilepath, mesh);
//     ASSERT(mesh.filepath == xfilepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::XDMF, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     um2::Vector<int8_t> element_types = {5, 5};
//     ASSERT(mesh.element_types == element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(tri_quad_mesh)
//     um2::MeshFile<T, I> mesh_ref;
//     make_tri_quad_reference_mesh_file(mesh_ref);
//     mesh_ref.filepath = std::filesystem::current_path().string() +
//     "/xdmf_test/tri_quad.inp"; um2::write_xdmf_file<T, I>(mesh_ref); std::string
//     xfilepath =
//         std::filesystem::current_path().string() + "/xdmf_test/tri_quad.xdmf";
//     um2::MeshFile<T, I> mesh;
//     um2::read_xdmf_file(xfilepath, mesh);
//     ASSERT(mesh.filepath == xfilepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::XDMF, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     um2::Vector<int8_t> element_types = {5, 4};
//     ASSERT(mesh.element_types == element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(tri6_mesh)
//     um2::MeshFile<T, I> mesh_ref;
//     make_tri6_reference_mesh_file(mesh_ref);
//     mesh_ref.filepath = std::filesystem::current_path().string() +
//     "/xdmf_test/tri6.inp"; um2::write_xdmf_file<T, I>(mesh_ref); std::string xfilepath
//     =
//         std::filesystem::current_path().string() + "/xdmf_test/tri6.xdmf";
//     um2::MeshFile<T, I> mesh;
//     um2::read_xdmf_file(xfilepath, mesh);
//     ASSERT(mesh.filepath == xfilepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::XDMF, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     um2::Vector<int8_t> element_types = {36, 36};
//     ASSERT(mesh.element_types == element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(quad8_mesh)
//     um2::MeshFile<T, I> mesh_ref;
//     make_quad8_reference_mesh_file(mesh_ref);
//     mesh_ref.filepath = std::filesystem::current_path().string() +
//     "/xdmf_test/quad8.inp"; um2::write_xdmf_file<T, I>(mesh_ref); std::string xfilepath
//     =
//         std::filesystem::current_path().string() + "/xdmf_test/quad8.xdmf";
//     um2::MeshFile<T, I> mesh;
//     um2::read_xdmf_file(xfilepath, mesh);
//     ASSERT(mesh.filepath == xfilepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::XDMF, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     um2::Vector<int8_t> element_types = {37, 37};
//     ASSERT(mesh.element_types == element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(tri6_quad8_mesh)
//     um2::MeshFile<T, I> mesh_ref;
//     make_tri6_quad8_reference_mesh_file(mesh_ref);
//     mesh_ref.filepath = std::filesystem::current_path().string() +
//     "/xdmf_test/tri6_quad8.inp"; um2::write_xdmf_file<T, I>(mesh_ref); std::string
//     xfilepath =
//         std::filesystem::current_path().string() + "/xdmf_test/tri6_quad8.xdmf";
//     um2::MeshFile<T, I> mesh;
//     um2::read_xdmf_file(xfilepath, mesh);
//     ASSERT(mesh.filepath == xfilepath, "filepath");
//     ASSERT(mesh.format == um2::MeshFileFormat::XDMF, "format");
//     ASSERT(mesh.name == mesh_ref.name, "name");
//     ASSERT(um2::compare_geometry(mesh, mesh_ref) == 0, "geometry");
//     um2::Vector<int8_t> element_types = {37, 36};
//     ASSERT(mesh.element_types == element_types, "element types");
//     ASSERT(mesh.elset_names == mesh_ref.elset_names, "elset names");
//     ASSERT(mesh.elset_offsets == mesh_ref.elset_offsets, "elset offsets");
//     ASSERT(mesh.elset_ids == mesh_ref.elset_ids, "elset ids");
// END_TEST

template <std::floating_point T, std::integral I>
TEST_SUITE(io_xdmf)
{
  TEST((tri_mesh<T, I>));
  // TEST((quad_mesh<T, I>) );
  // TEST(h", (tri_quad_mesh<T, I>) );
  // TEST((tri6_mesh<T, I>) );
  // TEST( (quad8_mesh<T, I>) );
  // TEST(esh", (tri6_quad8_mesh<T, I>) );
}

auto
main() -> int
{
  //    namespace fs = std::filesystem;
  //    std::string dir = fs::current_path().string() + "/xdmf_test";
  //    bool const success = fs::create_directory(dir);
  //    if (!success) {
  //        std::cerr << "Failed to create test directory: " << dir << std::endl;
  //        return 1;
  //    }
  RUN_SUITE((io_xdmf<float, int16_t>));
  RUN_SUITE((io_xdmf<float, int32_t>));
  RUN_SUITE((io_xdmf<float, int64_t>));
  RUN_SUITE((io_xdmf<double, int16_t>));
  RUN_SUITE((io_xdmf<double, int32_t>));
  RUN_SUITE((io_xdmf<double, int64_t>));
  //    fs::remove_all(dir);

  return 0;
}
