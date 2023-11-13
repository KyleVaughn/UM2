#include <um2/mesh/polytope_soup.hpp>

#include "./helpers/setup_polytope_soup.hpp"

#include "../test_macros.hpp"

template <std::floating_point T, std::signed_integral I>
TEST_CASE(addVertex)
{
  um2::PolytopeSoup<T, I> soup;
  ASSERT(soup.addVertex(1, 2, 3) == 0);
  ASSERT(soup.addVertex(2, 3, 4) == 1);

  um2::Point3<T> const p0 = soup.getVertex(0);
  ASSERT(um2::isApprox(p0, um2::Point3<T>(1, 2, 3)));
  um2::Point3<T> const p1 = soup.getVertex(1);
  ASSERT(um2::isApprox(p1, um2::Point3<T>(2, 3, 4)));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(addElement)
{
  um2::PolytopeSoup<T, I> soup;
  um2::Vector<I> conn = {0};
  ASSERT(soup.addVertex(0, 0) == 0);
  ASSERT(soup.addVertex(1, 0) == 1);
  ASSERT(soup.addVertex(0, 1) == 2);
  ASSERT(soup.addElement(um2::VTKElemType::Vertex, conn) == 0);
  conn = {0, 1};
  ASSERT(soup.addElement(um2::VTKElemType::Line, conn) == 1);
  conn = {0, 1, 2};
  ASSERT(soup.addElement(um2::VTKElemType::Triangle, conn) == 2);

  um2::VTKElemType elem_type= um2::VTKElemType::None;
  soup.getElement(0, elem_type, conn);
  ASSERT(elem_type == um2::VTKElemType::Vertex);
  um2::Vector<I> conn_ref = {0};
  ASSERT(conn == conn_ref);

  soup.getElement(1, elem_type, conn);
  ASSERT(elem_type == um2::VTKElemType::Line);
  conn_ref = {0, 1};
  ASSERT(conn == conn_ref);

  soup.getElement(2, elem_type, conn);
  ASSERT(elem_type == um2::VTKElemType::Triangle);
  conn_ref = {0, 1, 2};
  ASSERT(conn == conn_ref);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(addElset)
{
  um2::PolytopeSoup<T, I> soup;
  soup.addVertex(0, 0);
  soup.addVertex(1, 0);
  soup.addVertex(0, 1);
  soup.addVertex(1, 1);
  um2::Vector<I> conn = {0};
  soup.addElement(um2::VTKElemType::Vertex, conn);
  conn = {0, 1};
  soup.addElement(um2::VTKElemType::Line, conn);
  conn = {0, 1, 2};
  soup.addElement(um2::VTKElemType::Triangle, conn);
  conn = {0, 1, 2, 3};
  soup.addElement(um2::VTKElemType::Quad, conn);

  soup.addElset("all", {0, 1, 2, 3}, {11, 12, 13, 14});
  soup.addElset("tri", {2});
  ASSERT(soup.numElsets() == 2);

  um2::String name;
  um2::Vector<I> ids;
  um2::Vector<T> elset_data;
  soup.getElset(0, name, ids, elset_data);
  ASSERT(name == "all");
  ASSERT(ids == um2::Vector<I>({0, 1, 2, 3}));
  ASSERT(elset_data == um2::Vector<T>({11, 12, 13, 14}));
  elset_data.clear();
  soup.getElset(1, name, ids, elset_data);
  ASSERT(name == "tri");
  ASSERT(ids == um2::Vector<I>({2}));
  ASSERT(elset_data.empty());
}

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

template <std::floating_point T, std::signed_integral I>
TEST_CASE(getMeshType)
{
  um2::PolytopeSoup<T, I> tri;
  makeReferenceTriPolytopeSoup(tri);
  ASSERT(tri.getMeshType() == um2::MeshType::Tri);

  um2::PolytopeSoup<T, I> quad;
  makeReferenceQuadPolytopeSoup(quad);
  ASSERT(quad.getMeshType() == um2::MeshType::Quad);

  um2::PolytopeSoup<T, I> tri_quad;
  makeReferenceTriQuadPolytopeSoup(tri_quad);
  ASSERT(tri_quad.getMeshType() == um2::MeshType::TriQuad);

  um2::PolytopeSoup<T, I> tri6;
  makeReferenceTri6PolytopeSoup(tri6);
  ASSERT(tri6.getMeshType() == um2::MeshType::QuadraticTri);

  um2::PolytopeSoup<T, I> quad8;
  makeReferenceQuad8PolytopeSoup(quad8);
  ASSERT(quad8.getMeshType() == um2::MeshType::QuadraticQuad);

  um2::PolytopeSoup<T, I> tri6_quad8;
  makeReferenceTri6Quad8PolytopeSoup(tri6_quad8);
  ASSERT(tri6_quad8.getMeshType() == um2::MeshType::QuadraticTriQuad);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(sortElsets)
{
  um2::PolytopeSoup<T, I> tri;
  um2::PolytopeSoup<T, I> tri_ref;

  tri.addVertex(0, 0);
  tri.addVertex(1, 0);
  tri.addVertex(0, 1);
  tri.addVertex(1, 1);
  tri_ref.addVertex(0, 0);
  tri_ref.addVertex(1, 0);
  tri_ref.addVertex(0, 1);
  tri_ref.addVertex(1, 1);

  um2::Vector<I> conn = {0, 1, 2};
  tri.addElement(um2::VTKElemType::Triangle, conn);
  tri_ref.addElement(um2::VTKElemType::Triangle, conn);
  conn = {1, 3, 2};
  tri.addElement(um2::VTKElemType::Triangle, conn);
  tri_ref.addElement(um2::VTKElemType::Triangle, conn);

  tri_ref.addElset("A", {0, 1}, {10, 2});
  tri_ref.addElset("B", {1});
  tri_ref.addElset("Material_H2O", {1});
  tri_ref.addElset("Material_UO2", {0});

  tri.addElset("Material_H2O", {1});
  tri.addElset("B", {1});
  tri.addElset("Material_UO2", {0});
  tri.addElset("A", {0, 1}, {10, 2});

  tri.sortElsets();

  um2::String name;
  um2::Vector<I> ids;
  um2::Vector<T> elset_data;
  tri.getElset(0, name, ids, elset_data);
  ASSERT(name == "A");
  ASSERT(ids == um2::Vector<I>({0, 1}));
  ASSERT(elset_data == um2::Vector<T>({10, 2}));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(getSubmesh)
{
  um2::PolytopeSoup<T, I> tri_quad;
  makeReferenceTriQuadPolytopeSoup(tri_quad);
  um2::PolytopeSoup<T, I> tri_quad_a;

  tri_quad.getSubmesh("A", tri_quad_a);
  ASSERT(tri_quad.compareTo(tri_quad_a) == 10);
  um2::String name;
  um2::Vector<I> ids;
  um2::Vector<T> elset_data;
  tri_quad_a.getElset(0, name, ids, elset_data);
  ASSERT(name == "B");
  ASSERT(ids == um2::Vector<I>({1}));
  ASSERT(elset_data.empty());
  tri_quad_a.getElset(1, name, ids, elset_data);
  ASSERT(name == "Material_H2O");
  ASSERT(ids == um2::Vector<I>({1}));
  ASSERT(elset_data.empty());
  tri_quad_a.getElset(2, name, ids, elset_data);
  ASSERT(name == "Material_UO2");
  ASSERT(ids == um2::Vector<I>({0}));
  ASSERT(elset_data.empty());

  um2::PolytopeSoup<T, I> tri_quad_h2o;
  tri_quad.getSubmesh("Material_H2O", tri_quad_h2o);

  // (1,0), (1,1), (2,0)
  ASSERT(tri_quad_h2o.numVerts() == 3);
  ASSERT(um2::isApprox(tri_quad_h2o.getVertex(0), um2::Point3<T>(1, 0, 0)));
  ASSERT(um2::isApprox(tri_quad_h2o.getVertex(1), um2::Point3<T>(1, 1, 0)));
  ASSERT(um2::isApprox(tri_quad_h2o.getVertex(2), um2::Point3<T>(2, 0, 0)));

  ASSERT(tri_quad_h2o.numElems() == 1);
  um2::VTKElemType elem_type = um2::VTKElemType::None;
  um2::Vector<I> conn;
  tri_quad_h2o.getElement(0, elem_type, conn);
  ASSERT(elem_type == um2::VTKElemType::Triangle);
  ASSERT(conn == um2::Vector<I>({0, 2, 1}));
  
  ASSERT(tri_quad_h2o.numElsets() == 2);
  tri_quad_h2o.getElset(0, name, ids, elset_data);
  ASSERT(name == "A");
  ASSERT(ids == um2::Vector<I>({0}));
  ASSERT(elset_data.size() == 1);
  ASSERT_NEAR(elset_data[0], 2, static_cast<T>(1e-6));
  
  elset_data.clear();
  tri_quad_h2o.getElset(1, name, ids, elset_data);
  ASSERT(name == "B");
  ASSERT(ids == um2::Vector<I>({0}));
  ASSERT(elset_data.empty());
}

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

template <std::floating_point T, std::signed_integral I>
TEST_CASE(getMaterialIDs)
{
  um2::PolytopeSoup<T, I> tri_ref;
  makeReferenceTriPolytopeSoup(tri_ref);
  um2::Vector<MaterialID> mat_ids;
  tri_ref.getMaterialIDs(mat_ids, {"Material_H2O", "Material_UO2"});
  um2::Vector<MaterialID> const mat_ids_ref = {1, 0};
  ASSERT(mat_ids == mat_ids_ref);
  mat_ids.clear();
  tri_ref.getMaterialIDs(mat_ids, {"Material_UO2", "Material_H2O"});
  um2::Vector<MaterialID> const mat_ids_ref2 = {0, 1};
  ASSERT(mat_ids == mat_ids_ref2);
}

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(PolytopeSoup)
{
  TEST((addVertex<T, I>));
  TEST((addElement<T, I>));
  TEST((addElset<T, I>));
  TEST(verticesPerElem);
  TEST((getMeshType<T, I>));
  TEST((sortElsets<T, I>));
  TEST((getSubmesh<T, I>));
  TEST((getMaterialNames<T, I>));
  TEST((getMaterialIDs<T, I>));
}

auto
main() -> int
{
//  RUN_SUITE((PolytopeSoup<float, int16_t>));
//  RUN_SUITE((PolytopeSoup<float, int32_t>));
//  RUN_SUITE((PolytopeSoup<float, int64_t>));
//  RUN_SUITE((PolytopeSoup<double, int16_t>));
//  RUN_SUITE((PolytopeSoup<double, int32_t>));
  RUN_SUITE((PolytopeSoup<double, int64_t>));
  return 0;
}
