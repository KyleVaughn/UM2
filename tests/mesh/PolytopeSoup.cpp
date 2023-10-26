#include <um2/mesh/PolytopeSoup.hpp>

#include "./helpers/setup_polytope_soup.hpp"

#include "../test_macros.hpp"

TEST_CASE(verticesPerElem)
{
  static_assert(um2::verticesPerElem(um2::VTKElemType::Vertex) == 1);
  static_assert(um2::verticesPerElem(um2::VTKElemType::Line) == 2);
  static_assert(um2::verticesPerElem(um2::VTKElemType::Triangle) == 3);
  static_assert(um2::verticesPerElem(um2::VTKElemType::Quad) == 4);
  static_assert(um2::verticesPerElem(um2::VTKElemType::QuadraticEdge) == 3);
  static_assert(um2::verticesPerElem(um2::VTKElemType::QuadraticTriangle) == 6);
  static_assert(um2::verticesPerElem(um2::VTKElemType::QuadraticQuad) == 8);
}

// template <std::floating_point T, std::signed_integral I>
// TEST_CASE(getMeshType)
//{
//   um2::PolytopeSoup<T, I> tri;
//   makeReferenceTriPolytopeSoup(tri);
//   ASSERT(tri.getMeshType() == um2::MeshType::Tri);
//
//   um2::PolytopeSoup<T, I> quad;
//   makeReferenceQuadPolytopeSoup(quad);
//   ASSERT(quad.getMeshType() == um2::MeshType::Quad);
//
//   um2::PolytopeSoup<T, I> tri_quad;
//   makeReferenceTriQuadPolytopeSoup(tri_quad);
//   ASSERT(tri_quad.getMeshType() == um2::MeshType::TriQuad);
//
//   um2::PolytopeSoup<T, I> tri6;
//   makeReferenceTri6PolytopeSoup(tri6);
//   ASSERT(tri6.getMeshType() == um2::MeshType::QuadraticTri);
//
//   um2::PolytopeSoup<T, I> quad8;
//   makeReferenceQuad8PolytopeSoup(quad8);
//   ASSERT(quad8.getMeshType() == um2::MeshType::QuadraticQuad);
//
//   um2::PolytopeSoup<T, I> tri6_quad8;
//   makeReferenceTri6Quad8PolytopeSoup(tri6_quad8);
//   ASSERT(tri6_quad8.getMeshType() == um2::MeshType::QuadraticTriQuad);
// }
//
template <std::floating_point T, std::signed_integral I>
TEST_CASE(compareGeometry)
{
  um2::PolytopeSoup<T, I> tri_ref;
  makeReferenceTriPolytopeSoup(tri_ref);
  um2::PolytopeSoup<T, I> tri;
  makeReferenceTriPolytopeSoup(tri);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.vertices.push_back(um2::Point3<T>{0, 0, 0});
  ASSERT(um2::compareGeometry(tri, tri_ref) == 1);
  tri.vertices.pop_back();
  T const eps = um2::eps_distance<T>;
  tri.vertices[0][0] += (eps / 2);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.vertices[0][0] += (eps * 2);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 2);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(compareTopology)
{
  um2::PolytopeSoup<T, I> tri_ref;
  makeReferenceTriPolytopeSoup(tri_ref);
  um2::PolytopeSoup<T, I> tri;
  makeReferenceTriPolytopeSoup(tri);
  ASSERT(um2::compareGeometry(tri, tri_ref) == 0);
  tri.element_types.push_back(um2::VTKElemType::Quad);
  ASSERT(um2::compareTopology(tri, tri_ref) == 1);
  tri.element_types.pop_back();
  tri.element_types[0] = um2::VTKElemType::Quad;
  ASSERT(um2::compareTopology(tri, tri_ref) == 2);
  tri.element_types[0] = um2::VTKElemType::Triangle;
  tri.element_conn[0] += 1;
  ASSERT(um2::compareTopology(tri, tri_ref) == 3);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(sortElsets)
{
  um2::PolytopeSoup<T, I> tri_ref;
  makeReferenceTriPolytopeSoup(tri_ref);
  um2::PolytopeSoup<T, I> tri;
  makeReferenceTriPolytopeSoup(tri);
  // Reorder elsets
  tri.elset_names = {"Material_UO2", "Material_H2O", "A", "B"};
  tri.elset_offsets = {0, 1, 2, 4, 5};
  tri.elset_ids = {0, 1, 0, 1, 1};
  tri.sortElsets();
  ASSERT(tri.elset_names == tri_ref.elset_names);
  ASSERT(tri.elset_offsets == tri_ref.elset_offsets);
  ASSERT(tri.elset_ids == tri_ref.elset_ids);
}

// template <std::floating_point T, std::signed_integral I>
// TEST_CASE(getSubmesh)
//{
//   um2::PolytopeSoup<T, I> tri_quad;
//   makeReferenceTriQuadPolytopeSoup(tri_quad);
//   um2::PolytopeSoup<T, I> tri_quad_a;
//
//   tri_quad.getSubmesh("A", tri_quad_a);
//   ASSERT(tri_quad_a.filepath == "");
//   ASSERT(tri_quad_a.name == "A");
//   ASSERT(um2::compareGeometry(tri_quad_a, tri_quad_a) == 0);
//   ASSERT(um2::compareTopology(tri_quad_a, tri_quad_a) == 0);
//   ASSERT(tri_quad_a.elset_names.size() == 3);
//   ASSERT(tri_quad_a.elset_names[0] == "B");
//   ASSERT(tri_quad_a.elset_names[1] == "Material_H2O");
//   ASSERT(tri_quad_a.elset_names[2] == "Material_UO2");
//   std::vector<I> const elset_offsets_ref = {0, 1, 2, 3};
//   ASSERT(tri_quad_a.elset_offsets == elset_offsets_ref);
//   std::vector<I> const elset_ids_ref = {1, 1, 0};
//   ASSERT(tri_quad_a.elset_ids == elset_ids_ref);
//
//   um2::PolytopeSoup<T, I> tri_quad_h2o;
//   tri_quad.getSubmesh("Material_H2O", tri_quad_h2o);
//   ASSERT(tri_quad_h2o.filepath == "");
//   ASSERT(tri_quad_h2o.name == "Material_H2O");
//   ASSERT(tri_quad_h2o.vertices.size() == 3);
//   // (1,0), (2,0), (1,1)
//   ASSERT_NEAR(tri_quad_h2o.vertices[0][0], static_cast<T>(1), static_cast<T>(1e-6));
//   ASSERT_NEAR(tri_quad_h2o.vertices[0][1], static_cast<T>(0), static_cast<T>(1e-6));
//   ASSERT_NEAR(tri_quad_h2o.vertices[0][2], static_cast<T>(0), static_cast<T>(1e-6));
//   ASSERT_NEAR(tri_quad_h2o.vertices[1][0], static_cast<T>(1), static_cast<T>(1e-6));
//   ASSERT_NEAR(tri_quad_h2o.vertices[1][1], static_cast<T>(1), static_cast<T>(1e-6));
//   ASSERT_NEAR(tri_quad_h2o.vertices[1][2], static_cast<T>(0), static_cast<T>(1e-6));
//   ASSERT_NEAR(tri_quad_h2o.vertices[2][0], static_cast<T>(2), static_cast<T>(1e-6));
//   ASSERT_NEAR(tri_quad_h2o.vertices[2][1], static_cast<T>(0), static_cast<T>(1e-6));
//   ASSERT_NEAR(tri_quad_h2o.vertices[2][2], static_cast<T>(0), static_cast<T>(1e-6));
//   ASSERT(tri_quad_h2o.element_conn[0] == 0);
//   ASSERT(tri_quad_h2o.element_conn[1] == 2);
//   ASSERT(tri_quad_h2o.element_conn[2] == 1);
//   ASSERT(tri_quad_h2o.elset_names.size() == 2);
//   ASSERT(tri_quad_h2o.elset_names[0] == "A");
//   ASSERT(tri_quad_h2o.elset_names[1] == "B");
//   ASSERT(tri_quad_h2o.elset_offsets.size() == 3);
//   ASSERT(tri_quad_h2o.elset_offsets[0] == 0);
//   ASSERT(tri_quad_h2o.elset_offsets[1] == 1);
//   ASSERT(tri_quad_h2o.elset_offsets[2] == 2);
//   ASSERT(tri_quad_h2o.elset_ids.size() == 2);
//   ASSERT(tri_quad_h2o.elset_ids[0] == 0);
//   ASSERT(tri_quad_h2o.elset_ids[1] == 0);
// }
//
template <std::floating_point T, std::signed_integral I>
TEST_CASE(getMaterialNames)
{
  um2::PolytopeSoup<T, I> tri_ref;
  makeReferenceTriPolytopeSoup(tri_ref);
  um2::Vector<um2::String> const mat_names_ref = {"Material_H2O", "Material_UO2"};
  um2::Vector<um2::String> mat_names;
  tri_ref.getMaterialNames(mat_names);
  ASSERT(mat_names == mat_names_ref);
}

// template <std::floating_point T, std::signed_integral I>
// TEST_CASE(getMaterialIDs)
//{
//   um2::PolytopeSoup<T, I> tri_ref;
//   makeReferenceTriPolytopeSoup(tri_ref);
//   std::vector<MaterialID> mat_ids;
//   tri_ref.getMaterialIDs(mat_ids,
//                          std::vector<std::string>{"Material_H2O", "Material_UO2"});
//   std::vector<MaterialID> const mat_ids_ref = {1, 0};
//   ASSERT(mat_ids == mat_ids_ref);
//   mat_ids.clear();
//   tri_ref.getMaterialIDs(mat_ids,
//                          std::vector<std::string>{"Material_UO2", "Material_H2O"});
//   std::vector<MaterialID> const mat_ids_ref2 = {0, 1};
//   ASSERT(mat_ids == mat_ids_ref2);
// }

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(PolytopeSoup)
{
  TEST(verticesPerElem);
  ///  TEST((getMeshType<T, I>));
  TEST((compareGeometry<T, I>));
  TEST((compareTopology<T, I>));
  TEST((sortElsets<T, I>));
  ///  TEST((getSubmesh<T, I>));
  TEST((getMaterialNames<T, I>));
  ///  TEST((getMaterialIDs<T, I>));
}

auto
main() -> int
{
  RUN_SUITE((PolytopeSoup<float, int16_t>));
  RUN_SUITE((PolytopeSoup<float, int32_t>));
  RUN_SUITE((PolytopeSoup<float, int64_t>));
  RUN_SUITE((PolytopeSoup<double, int16_t>));
  RUN_SUITE((PolytopeSoup<double, int32_t>));
  RUN_SUITE((PolytopeSoup<double, int64_t>));
  return 0;
}
