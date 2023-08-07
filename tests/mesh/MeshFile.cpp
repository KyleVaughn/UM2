#include <um2/mesh/MeshFile.hpp>

#include "./helpers/setup_mesh_file.hpp"

#include "../test_macros.hpp"
#include "um2/common/Vector.hpp"

template <std::floating_point T, std::signed_integral I>
TEST_CASE(compare_geometry)
{
  //  std::string const filename = "./mesh_files/tri.inp";
  um2::MeshFile<T, I> tri_ref;
  makeReferenceTriMeshFile(tri_ref);
  um2::MeshFile<T, I> tri;
  makeReferenceTriMeshFile(tri);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  // Compare x,y,z
  // Trivial inequality x
  T node_tmp = tri.nodes_x[0];
  tri.nodes_x[0] = node_tmp + 1;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 1);
  tri.nodes_x[0] = node_tmp;
  // Trivial inequality y
  node_tmp = tri.nodes_y[0];
  tri.nodes_y[0] = node_tmp + 1;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 2);
  tri.nodes_y[0] = node_tmp;
  // Trivial inequality z
  node_tmp = tri.nodes_z[0];
  tri.nodes_z[0] = node_tmp + 1;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 3);
  tri.nodes_z[0] = node_tmp;
  // Non-trivial equality/inequality x
  T const eps = um2::epsilonDistance<T>();
  node_tmp = tri.nodes_x[0];
  tri.nodes_x[0] = node_tmp + 2 * eps;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 1);
  tri.nodes_x[0] = node_tmp + eps / 2;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.nodes_x[0] = node_tmp;
  // Non-trivial equality/inequality y
  node_tmp = tri.nodes_y[0];
  tri.nodes_y[0] = node_tmp + 2 * eps;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 2);
  tri.nodes_y[0] = node_tmp + eps / 2;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.nodes_y[0] = node_tmp;
  // Non-trivial equality/inequality z
  node_tmp = tri.nodes_z[0];
  tri.nodes_z[0] = node_tmp + 2 * eps;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 3);
  tri.nodes_z[0] = node_tmp + eps / 2;
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.nodes_z[0] = node_tmp;
  // Compare topology
  // element_offsets
  tri.element_offsets[0] = 1;
  ASSERT(um2::compareTopology(tri, tri_ref) == 1);
  tri.element_offsets[0] = 0;
  // element_conn
  tri.element_conn[0] = 1;
  ASSERT(um2::compareTopology(tri, tri_ref) == 2);
  tri.element_conn[0] = 0;
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(sort_elsets)
{
  um2::MeshFile<T, I> tri_ref;
  makeReferenceTriMeshFile(tri_ref);
  um2::MeshFile<T, I> tri;
  makeReferenceTriMeshFile(tri);
  // Reorder elsets
  tri.elset_names = {"Material_UO2", "Material_H2O", "A", "B"};
  tri.elset_offsets = {0, 1, 2, 4, 5};
  tri.elset_ids = {0, 1, 0, 1, 1};
  tri.sortElsets();
  ASSERT(tri.elset_names == tri_ref.elset_names);
  ASSERT(tri.elset_offsets == tri_ref.elset_offsets);
  ASSERT(tri.elset_ids == tri_ref.elset_ids);
}
//
// template <std::floating_point T, std::signed_integral I>
// TEST(get_submesh)
//     um2::MeshFile<T, I> tri_quad;
//     make_tri_quad_reference_mesh_file(tri_quad);
//     um2::MeshFile<T, I> tri_quad_A;
//
//     tri_quad.get_submesh("A", tri_quad_A);
//     ASSERT(tri_quad_A.filepath == "", "get_submesh");
//     ASSERT(tri_quad_A.name == "A", "get_submesh");
//     ASSERT(tri_quad_A.format == tri_quad.format, "get_submesh");
//     ASSERT(um2::is_approx(tri_quad_A.nodes_x, tri_quad.nodes_x), "get_submesh");
//     ASSERT(um2::is_approx(tri_quad_A.nodes_y, tri_quad.nodes_y), "get_submesh");
//     ASSERT(um2::is_approx(tri_quad_A.nodes_z, tri_quad.nodes_z), "get_submesh");
//     ASSERT(tri_quad_A.element_types == tri_quad.element_types, "get_submesh");
//     ASSERT(tri_quad_A.element_offsets == tri_quad.element_offsets, "get_submesh");
//     ASSERT(tri_quad_A.element_conn == tri_quad.element_conn, "get_submesh");
//     ASSERT(tri_quad_A.elset_names.size() == 3, "get_submesh");
//     ASSERT(tri_quad_A.elset_names[0] == "B", "get_submesh");
//     ASSERT(tri_quad_A.elset_names[1] == "Material_H2O", "get_submesh");
//     ASSERT(tri_quad_A.elset_names[2] == "Material_UO2", "get_submesh");
//     um2::Vector<I> elset_offsets_ref = {0, 1, 2, 3};
//     ASSERT(tri_quad_A.elset_offsets == elset_offsets_ref, "get_submesh");
//     um2::Vector<I> elset_ids_ref = {1, 1, 0};
//     ASSERT(tri_quad_A.elset_ids == elset_ids_ref, "get_submesh");
//
//     um2::MeshFile<T, I> tri_quad_UO2;
//     tri_quad.get_submesh("Material_UO2", tri_quad_UO2);
//     ASSERT(tri_quad_UO2.filepath == "", "get_submesh");
//     ASSERT(tri_quad_UO2.name == "Material_UO2", "get_submesh");
//     ASSERT(tri_quad_UO2.format == tri_quad.format, "get_submesh");
//     ASSERT(tri_quad_UO2.nodes_x.size() == 4, "get_submesh");
//     ASSERT(tri_quad_UO2.nodes_y.size() == 4, "get_submesh");
//     ASSERT(tri_quad_UO2.nodes_z.size() == 4, "get_submesh");
//     for (length_t i = 0; i < 4; ++i) {
//         ASSERT_APPROX(tri_quad_UO2.nodes_x[i], tri_quad.nodes_x[i], 1e-4,
//         "get_submesh"); ASSERT_APPROX(tri_quad_UO2.nodes_y[i], tri_quad.nodes_y[i],
//         1e-4, "get_submesh"); ASSERT_APPROX(tri_quad_UO2.nodes_z[i],
//         tri_quad.nodes_z[i], 1e-4, "get_submesh");
//     }
//     ASSERT(tri_quad_UO2.element_types.size() == 1, "get_submesh");
//     ASSERT(tri_quad_UO2.element_types[0] == 9, "get_submesh");
//     ASSERT(tri_quad_UO2.element_offsets.size() == 2, "get_submesh");
//     ASSERT(tri_quad_UO2.element_offsets[0] == 0, "get_submesh");
//     ASSERT(tri_quad_UO2.element_offsets[1] == 4, "get_submesh");
//     ASSERT(tri_quad_UO2.element_conn.size() == 4, "get_submesh");
//     ASSERT(tri_quad_UO2.element_conn[0] == 0, "get_submesh");
//     ASSERT(tri_quad_UO2.element_conn[1] == 1, "get_submesh");
//     ASSERT(tri_quad_UO2.element_conn[2] == 2, "get_submesh");
//     ASSERT(tri_quad_UO2.element_conn[3] == 3, "get_submesh");
//     ASSERT(tri_quad_UO2.elset_names.size() == 1, "get_submesh");
//     ASSERT(tri_quad_UO2.elset_names[0] == "A", "get_submesh");
//     ASSERT(tri_quad_UO2.elset_offsets.size() == 2, "get_submesh");
//     ASSERT(tri_quad_UO2.elset_offsets[0] == 0, "get_submesh");
//     ASSERT(tri_quad_UO2.elset_offsets[1] == 1, "get_submesh");
//     ASSERT(tri_quad_UO2.elset_ids.size() == 1, "get_submesh");
//     ASSERT(tri_quad_UO2.elset_ids[0] == 0, "get_submesh");
//
//     um2::MeshFile<T, I> tri_quad_H2O;
//     tri_quad.get_submesh("Material_H2O", tri_quad_H2O);
//     ASSERT(tri_quad_H2O.filepath == "", "get_submesh");
//     ASSERT(tri_quad_H2O.name == "Material_H2O", "get_submesh");
//     ASSERT(tri_quad_H2O.format == tri_quad.format, "get_submesh");
//     ASSERT(tri_quad_H2O.nodes_x.size() == 3, "get_submesh");
//     ASSERT(tri_quad_H2O.nodes_y.size() == 3, "get_submesh");
//     ASSERT(tri_quad_H2O.nodes_z.size() == 3, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_x[0], tri_quad.nodes_x[1], 1e-4, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_y[0], tri_quad.nodes_y[1], 1e-4, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_z[0], tri_quad.nodes_z[1], 1e-4, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_x[1], tri_quad.nodes_x[2], 1e-4, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_y[1], tri_quad.nodes_y[2], 1e-4, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_z[1], tri_quad.nodes_z[2], 1e-4, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_x[2], tri_quad.nodes_x[4], 1e-4, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_y[2], tri_quad.nodes_y[4], 1e-4, "get_submesh");
//     ASSERT_APPROX(tri_quad_H2O.nodes_z[2], tri_quad.nodes_z[4], 1e-4, "get_submesh");
//     ASSERT(tri_quad_H2O.element_types.size() == 1, "get_submesh");
//     ASSERT(tri_quad_H2O.element_types[0] == 5, "get_submesh");
//     ASSERT(tri_quad_H2O.element_offsets.size() == 2, "get_submesh");
//     ASSERT(tri_quad_H2O.element_offsets[0] == 0, "get_submesh");
//     ASSERT(tri_quad_H2O.element_offsets[1] == 3, "get_submesh");
//     ASSERT(tri_quad_H2O.element_conn.size() == 3, "get_submesh");
//     ASSERT(tri_quad_H2O.element_conn[0] == 0, "get_submesh");
//     ASSERT(tri_quad_H2O.element_conn[1] == 2, "get_submesh");
//     ASSERT(tri_quad_H2O.element_conn[2] == 1, "get_submesh");
//     ASSERT(tri_quad_H2O.elset_names.size() == 2, "get_submesh");
//     ASSERT(tri_quad_H2O.elset_names[0] == "A", "get_submesh");
//     ASSERT(tri_quad_H2O.elset_names[1] == "B", "get_submesh");
//     ASSERT(tri_quad_H2O.elset_offsets.size() == 3, "get_submesh");
//     ASSERT(tri_quad_H2O.elset_offsets[0] == 0, "get_submesh");
//     ASSERT(tri_quad_H2O.elset_offsets[1] == 1, "get_submesh");
//     ASSERT(tri_quad_H2O.elset_offsets[2] == 2, "get_submesh");
//     ASSERT(tri_quad_H2O.elset_ids.size() == 2, "get_submesh");
//     ASSERT(tri_quad_H2O.elset_ids[0] == 0, "get_submesh");
//     ASSERT(tri_quad_H2O.elset_ids[1] == 0, "get_submesh");
// END_TEST
//
// template <std::floating_point T, std::signed_integral I>
// TEST(get_mesh_type)
//     um2::MeshFile<T, I> tri_ref;
//     make_tri_reference_mesh_file(tri_ref);
//     um2::MeshType mesh_type = um2::MeshType::ERROR;
//     mesh_type = tri_ref.get_mesh_type();
//     ASSERT(mesh_type == um2::MeshType::TRI, "get_mesh_type");
//
//     um2::MeshFile<T, I> quad_ref;
//     make_quad_reference_mesh_file(quad_ref);
//     mesh_type = quad_ref.get_mesh_type();
//     ASSERT(mesh_type == um2::MeshType::QUAD, "get_mesh_type");
//
//     um2::MeshFile<T, I> tri_quad_ref;
//     make_tri_quad_reference_mesh_file(tri_quad_ref);
//     mesh_type = tri_quad_ref.get_mesh_type();
//     ASSERT(mesh_type == um2::MeshType::TRI_QUAD, "get_mesh_type");
//
//     um2::MeshFile<T, I> tri6_ref;
//     make_tri6_reference_mesh_file(tri6_ref);
//     mesh_type = tri6_ref.get_mesh_type();
//     ASSERT(mesh_type == um2::MeshType::QUADRATIC_TRI, "get_mesh_type");
//
//     um2::MeshFile<T, I> quad8_ref;
//     make_quad8_reference_mesh_file(quad8_ref);
//     mesh_type = quad8_ref.get_mesh_type();
//     ASSERT(mesh_type == um2::MeshType::QUADRATIC_QUAD, "get_mesh_type");
//
//     um2::MeshFile<T, I> tri6_quad8_ref;
//     make_tri6_quad8_reference_mesh_file(tri6_quad8_ref);
//     mesh_type = tri6_quad8_ref.get_mesh_type();
//     ASSERT(mesh_type == um2::MeshType::QUADRATIC_TRI_QUAD, "get_mesh_type");
// END_TEST

// template <std::floating_point T, std::signed_integral I>
// TEST_CASE(get_material_ids)
//{
//   um2::MeshFile<T, I> tri_ref;
//   makeReferenceTriMeshFile((tri_ref));
//   um2::Vector<MaterialID> mat_ids_ref = {1, 0};
//   um2::Vector<MaterialID> mat_ids;
//   tri_ref.getMaterialNames()
//   ASSERT(mat_ids == mat_ids_ref);
// }

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(MeshFile)
{
  TEST_HOSTDEV(compare_geometry, 1, 1, T, I)
  // compare tolopolgy
  TEST_HOSTDEV(sort_elsets, 1, 1, T, I);
  //    RUN_TEST("get_submesh", (get_submesh<T, I>) );
  //    RUN_TEST("get_mesh_type", (get_mesh_type<T, I>) );
  //  TEST_HOSTDEV(get_material_ids, 1, 1, T, I);
}

auto
main() -> int
{
  RUN_SUITE((MeshFile<float, int16_t>));
  RUN_SUITE((MeshFile<float, int32_t>));
  RUN_SUITE((MeshFile<float, int64_t>));
  RUN_SUITE((MeshFile<double, int16_t>));
  RUN_SUITE((MeshFile<double, int32_t>));
  RUN_SUITE((MeshFile<double, int64_t>));
  return 0;
}
