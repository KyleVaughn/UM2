#include <um2/mesh/MeshFile.hpp>

#include "./helpers/setup_mesh_file.hpp"

#include "../test_macros.hpp"

TEST_CASE(verticesPerCell)
{
  static_assert(um2::verticesPerCell(um2::MeshType::Tri) == 3);
  static_assert(um2::verticesPerCell(um2::MeshType::Quad) == 4);
  static_assert(um2::verticesPerCell(um2::MeshType::QuadraticTri) == 6);
  static_assert(um2::verticesPerCell(um2::MeshType::QuadraticQuad) == 8);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(compareGeometry)
{
  um2::MeshFile<T, I> tri_ref;
  makeReferenceTriMeshFile(tri_ref);
  um2::MeshFile<T, I> tri;
  makeReferenceTriMeshFile(tri);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.vertices.push_back(um2::Point3<T>{0, 0, 0});
  ASSERT(um2::compareGeometry(tri, tri_ref) == 1);
  tri.vertices.pop_back();
  T const eps = um2::epsilonDistance<T>();
  tri.vertices[0][0] += (eps / 2);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.vertices[0][0] += (eps * 2);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 2);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(compareTopology)
{
  um2::MeshFile<T, I> tri_ref;
  makeReferenceTriMeshFile(tri_ref);
  um2::MeshFile<T, I> tri;
  makeReferenceTriMeshFile(tri);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.type = um2::MeshType::Quad;
  ASSERT(um2::compareTopology(tri, tri_ref) == 1);
  tri.type = um2::MeshType::Tri;
  tri.element_conn.push_back(0);
  ASSERT(um2::compareTopology(tri, tri_ref) == 2);
  tri.element_conn.pop_back();
  tri.element_conn[0] += 1;
  ASSERT(um2::compareTopology(tri, tri_ref) == 3);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(sortElsets)
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

template <std::floating_point T, std::signed_integral I>
TEST_CASE(getSubmesh)
{
  um2::MeshFile<T, I> tri;
  makeReferenceTriMeshFile(tri);
  um2::MeshFile<T, I> tri_a;

  tri.getSubmesh("A", tri_a);
  ASSERT(tri_a.filepath == "");
  ASSERT(tri_a.name == "A");
  ASSERT(tri_a.format == tri.format);
  ASSERT(um2::compareGeometry(tri_a, tri) == 0);
  ASSERT(um2::compareTopology(tri_a, tri) == 0);
  ASSERT(tri_a.elset_names.size() == 3);
  ASSERT(tri_a.elset_names[0] == "B");
  ASSERT(tri_a.elset_names[1] == "Material_H2O");
  ASSERT(tri_a.elset_names[2] == "Material_UO2");
  std::vector<I> const elset_offsets_ref = {0, 1, 2, 3};
  ASSERT(tri_a.elset_offsets == elset_offsets_ref);
  std::vector<I> const elset_ids_ref = {1, 1, 0};
  ASSERT(tri_a.elset_ids == elset_ids_ref);

  um2::MeshFile<T, I> tri_h2o;
  tri.getSubmesh("Material_H2O", tri_h2o);
  ASSERT(tri_h2o.filepath == "");
  ASSERT(tri_h2o.name == "Material_H2O");
  ASSERT(tri_h2o.format == tri.format);
  ASSERT(tri_h2o.vertices.size() == 3);
  // (0,0), (1,1), (0,1)
  ASSERT_NEAR(tri_h2o.vertices[0][0], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(tri_h2o.vertices[0][1], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(tri_h2o.vertices[0][2], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(tri_h2o.vertices[1][0], static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(tri_h2o.vertices[1][1], static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(tri_h2o.vertices[1][2], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(tri_h2o.vertices[2][0], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(tri_h2o.vertices[2][1], static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(tri_h2o.vertices[2][2], static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT(tri_h2o.element_conn[0] == 1);
  ASSERT(tri_h2o.element_conn[1] == 2);
  ASSERT(tri_h2o.element_conn[2] == 0);
  ASSERT(tri_h2o.elset_names.size() == 2);
  ASSERT(tri_h2o.elset_names[0] == "A");
  ASSERT(tri_h2o.elset_names[1] == "B");
  ASSERT(tri_h2o.elset_offsets.size() == 3);
  ASSERT(tri_h2o.elset_offsets[0] == 0);
  ASSERT(tri_h2o.elset_offsets[1] == 1);
  ASSERT(tri_h2o.elset_offsets[2] == 2);
  ASSERT(tri_h2o.elset_ids.size() == 2);
  ASSERT(tri_h2o.elset_ids[0] == 0);
  ASSERT(tri_h2o.elset_ids[1] == 0);
}
//// template <std::floating_point T, std::signed_integral I>
//// TEST(get_material_ids)
////     um2::MeshFile<T, I> tri_ref;
////     make_tri_reference_mesh_file(tri_ref);
////     um2::Vector<MaterialID> mat_ids_ref = {1, 0};
////     um2::Vector<MaterialID> mat_ids;
////     tri_ref.get_material_ids(mat_ids);
////     ASSERT(mat_ids == mat_ids_ref, "get_material_ids");
//// END_TEST
////
template <std::floating_point T, std::signed_integral I>
TEST_SUITE(MeshFile)
{
  TEST(verticesPerCell);
  TEST((compareGeometry<T, I>));
  TEST((compareTopology<T, I>));
  TEST((sortElsets<T, I>));
  TEST((getSubmesh<T, I>));
  //    RUN_TEST("get_material_ids", (get_material_ids<T, I>) );
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
