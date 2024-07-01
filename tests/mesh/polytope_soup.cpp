#include <um2/common/cast_if_not.hpp>
#include <um2/config.hpp>
#include <um2/geometry/point.hpp>
#include <um2/mesh/element_types.hpp>
#include <um2/mesh/polytope_soup.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/stdlib/vector.hpp>

#include "./helpers/setup_polytope_soup.hpp"

#include "../test_macros.hpp"

#include <cstdio>

TEST_CASE(addVertex)
{
  um2::PolytopeSoup soup;
  ASSERT(soup.addVertex(1, 2, 3) == 0);
  ASSERT(soup.addVertex(2, 3, 4) == 1);

  um2::Point3F const p0 = soup.getVertex(0);
  ASSERT(p0.isApprox(um2::Point3F(1, 2, 3)));
  um2::Point3F const p1 = soup.getVertex(1);
  ASSERT(p1.isApprox(um2::Point3F(2, 3, 4)));
}

TEST_CASE(addElement)
{
  um2::PolytopeSoup soup;
  um2::Vector<Int> conn = {0};
  ASSERT(soup.addVertex(0, 0) == 0);
  ASSERT(soup.addVertex(1, 0) == 1);
  ASSERT(soup.addVertex(0, 1) == 2);
  ASSERT(soup.addElement(um2::VTKElemType::Vertex, conn) == 0);
  conn = {0, 1};
  ASSERT(soup.addElement(um2::VTKElemType::Line, conn) == 1);
  conn = {0, 1, 2};
  ASSERT(soup.addElement(um2::VTKElemType::Triangle, conn) == 2);

  um2::VTKElemType elem_type = um2::VTKElemType::Invalid;
  soup.getElement(0, elem_type, conn);
  ASSERT(elem_type == um2::VTKElemType::Vertex);
  um2::Vector<Int> conn_ref = {0};
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

TEST_CASE(addElset)
{
  um2::PolytopeSoup soup;
  soup.addVertex(0, 0);
  soup.addVertex(1, 0);
  soup.addVertex(0, 1);
  soup.addVertex(1, 1);
  um2::Vector<Int> conn = {0};
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
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  soup.getElset(0, name, ids, elset_data);
  ASSERT(name == "all");
  ASSERT(ids == um2::Vector<Int>({0, 1, 2, 3}));
  for (Int i = 0; i < 4; ++i) {
    ASSERT_NEAR(elset_data[i], castIfNot<Float>(11 + i), castIfNot<Float>(1e-6));
  }
  elset_data.clear();
  soup.getElset(1, name, ids, elset_data);
  ASSERT(name == "tri");
  ASSERT(ids == um2::Vector<Int>({2}));
  ASSERT(elset_data.empty());
}

TEST_CASE(sortElsets)
{
  um2::PolytopeSoup tri;
  um2::PolytopeSoup tri_ref;

  tri.addVertex(0, 0);
  tri.addVertex(1, 0);
  tri.addVertex(0, 1);
  tri.addVertex(1, 1);
  tri_ref.addVertex(0, 0);
  tri_ref.addVertex(1, 0);
  tri_ref.addVertex(0, 1);
  tri_ref.addVertex(1, 1);

  um2::Vector<Int> conn = {0, 1, 2};
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
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  tri.getElset(0, name, ids, elset_data);
  ASSERT(name == "A");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.size() == 2);
  ASSERT_NEAR(elset_data[0], 10, castIfNot<Float>(1e-6));
  ASSERT_NEAR(elset_data[1], 2, castIfNot<Float>(1e-6));
}

TEST_CASE(getSubset)
{
  um2::PolytopeSoup tri_quad;
  makeReferenceTriQuadPolytopeSoup(tri_quad);
  um2::PolytopeSoup tri_quad_a;

  tri_quad.getSubset("A", tri_quad_a);
  ASSERT(tri_quad.compare(tri_quad_a) == 6);
  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  tri_quad_a.getElset(0, name, ids, elset_data);
  ASSERT(name == "B");
  ASSERT(ids == um2::Vector<Int>({1}));
  ASSERT(elset_data.empty());
  tri_quad_a.getElset(1, name, ids, elset_data);
  ASSERT(name == "Material_H2O");
  ASSERT(ids == um2::Vector<Int>({1}));
  ASSERT(elset_data.empty());
  tri_quad_a.getElset(2, name, ids, elset_data);
  ASSERT(name == "Material_UO2");
  ASSERT(ids == um2::Vector<Int>({0}));
  ASSERT(elset_data.empty());

  um2::PolytopeSoup tri_quad_h2o;
  tri_quad.getSubset("Material_H2O", tri_quad_h2o);

  // (1,0), (1,1), (2,0)
  ASSERT(tri_quad_h2o.numVertices() == 3);
  ASSERT(tri_quad_h2o.getVertex(0).isApprox(um2::Point3F(1, 0, 0)));
  ASSERT(tri_quad_h2o.getVertex(1).isApprox(um2::Point3F(1, 1, 0)));
  ASSERT(tri_quad_h2o.getVertex(2).isApprox(um2::Point3F(2, 0, 0)));

  ASSERT(tri_quad_h2o.numElements() == 1);
  um2::VTKElemType elem_type = um2::VTKElemType::Invalid;
  um2::Vector<Int> conn;
  tri_quad_h2o.getElement(0, elem_type, conn);
  ASSERT(elem_type == um2::VTKElemType::Triangle);
  ASSERT(conn == um2::Vector<Int>({0, 2, 1}));

  ASSERT(tri_quad_h2o.numElsets() == 2);
  tri_quad_h2o.getElset(0, name, ids, elset_data);
  ASSERT(name == "A");
  ASSERT(ids == um2::Vector<Int>({0}));
  ASSERT(elset_data.size() == 1);
  ASSERT_NEAR(elset_data[0], 2, castIfNot<Float>(1e-6));

  elset_data.clear();
  tri_quad_h2o.getElset(1, name, ids, elset_data);
  ASSERT(name == "B");
  ASSERT(ids == um2::Vector<Int>({0}));
  ASSERT(elset_data.empty());
}

TEST_CASE(operator_plus_equal)
{
  um2::PolytopeSoup soup;
  soup.addVertex(0, 0);
  soup.addVertex(1, 0);
  soup.addVertex(1, 1);
  soup.addVertex(0, 1);
  soup.addElement(um2::VTKElemType::Triangle, {0, 1, 3});
  soup.addElement(um2::VTKElemType::Triangle, {1, 2, 3});
  soup.addElset("A", {0, 1});
  soup.addElset("C", {1}, {11});

  um2::PolytopeSoup soup2;
  soup2.addVertex(1, 0);
  soup2.addVertex(2, 0);
  soup2.addVertex(3, 0);
  soup2.addVertex(2, 1);
  soup2.addVertex(1, 1);
  soup2.addElement(um2::VTKElemType::Quad, {0, 1, 3, 4});
  soup2.addElement(um2::VTKElemType::Triangle, {1, 2, 3});
  soup2.addElset("B", {0, 1});
  soup2.addElset("C", {0, 1}, {22, 33});

  soup += soup2;
  // Check vertices
  ASSERT(soup.numVertices() == 9);
  ASSERT(soup.getVertex(0).isApprox(um2::Point3F(0, 0, 0)));
  ASSERT(soup.getVertex(1).isApprox(um2::Point3F(1, 0, 0)));
  ASSERT(soup.getVertex(2).isApprox(um2::Point3F(1, 1, 0)));
  ASSERT(soup.getVertex(3).isApprox(um2::Point3F(0, 1, 0)));
  ASSERT(soup.getVertex(4).isApprox(um2::Point3F(1, 0, 0)));
  ASSERT(soup.getVertex(5).isApprox(um2::Point3F(2, 0, 0)));
  ASSERT(soup.getVertex(6).isApprox(um2::Point3F(3, 0, 0)));
  ASSERT(soup.getVertex(7).isApprox(um2::Point3F(2, 1, 0)));
  ASSERT(soup.getVertex(8).isApprox(um2::Point3F(1, 1, 0)));
  // Check elements
  ASSERT(soup.numElements() == 4);
  um2::Vector<Int> conn;
  um2::VTKElemType type = um2::VTKElemType::Invalid;

  soup.getElement(0, type, conn);
  ASSERT(type == um2::VTKElemType::Triangle);
  ASSERT(conn == um2::Vector<Int>({0, 1, 3}));
  type = um2::VTKElemType::Invalid;
  conn.clear();

  soup.getElement(1, type, conn);
  ASSERT(type == um2::VTKElemType::Triangle);
  ASSERT(conn == um2::Vector<Int>({1, 2, 3}));
  type = um2::VTKElemType::Invalid;
  conn.clear();

  soup.getElement(2, type, conn);
  ASSERT(type == um2::VTKElemType::Quad);
  ASSERT(conn == um2::Vector<Int>({4, 5, 7, 8}));
  type = um2::VTKElemType::Invalid;
  conn.clear();

  soup.getElement(3, type, conn);
  ASSERT(type == um2::VTKElemType::Triangle);
  ASSERT(conn == um2::Vector<Int>({5, 6, 7}));

  // Check elsets
  ASSERT(soup.numElsets() == 3);
  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  soup.getElset(0, name, ids, elset_data);
  ASSERT(name == "A");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.empty());
  ids.clear();
  elset_data.clear();

  soup.getElset(1, name, ids, elset_data);
  ASSERT(name == "B");
  ASSERT(ids == um2::Vector<Int>({2, 3}));
  ASSERT(elset_data.empty());
  ids.clear();
  elset_data.clear();

  soup.getElset(2, name, ids, elset_data);
  ASSERT(name == "C");
  ASSERT(ids == um2::Vector<Int>({1, 2, 3}));
  ASSERT(elset_data.size() == 3);
  auto constexpr eps = castIfNot<Float>(1e-6);
  ASSERT_NEAR(elset_data[0], 11, eps);
  ASSERT_NEAR(elset_data[1], 22, eps);
  ASSERT_NEAR(elset_data[2], 33, eps);
  ids.clear();
  elset_data.clear();

  um2::PolytopeSoup soup3;
  soup3 += soup;
  // Check vertices
  ASSERT(soup3.numVertices() == 9);
  ASSERT(soup3.getVertex(0).isApprox(um2::Point3F(0, 0, 0)));
  ASSERT(soup3.getVertex(1).isApprox(um2::Point3F(1, 0, 0)));
  ASSERT(soup3.getVertex(2).isApprox(um2::Point3F(1, 1, 0)));
  ASSERT(soup3.getVertex(3).isApprox(um2::Point3F(0, 1, 0)));
  ASSERT(soup3.getVertex(4).isApprox(um2::Point3F(1, 0, 0)));
  ASSERT(soup3.getVertex(5).isApprox(um2::Point3F(2, 0, 0)));
  ASSERT(soup3.getVertex(6).isApprox(um2::Point3F(3, 0, 0)));
  ASSERT(soup3.getVertex(7).isApprox(um2::Point3F(2, 1, 0)));
  ASSERT(soup3.getVertex(8).isApprox(um2::Point3F(1, 1, 0)));
  // Check elements
  ASSERT(soup3.numElements() == 4);

  soup3.getElement(0, type, conn);
  ASSERT(type == um2::VTKElemType::Triangle);
  ASSERT(conn == um2::Vector<Int>({0, 1, 3}));
  type = um2::VTKElemType::Invalid;
  conn.clear();

  soup3.getElement(1, type, conn);
  ASSERT(type == um2::VTKElemType::Triangle);
  ASSERT(conn == um2::Vector<Int>({1, 2, 3}));
  type = um2::VTKElemType::Invalid;
  conn.clear();

  soup3.getElement(2, type, conn);
  ASSERT(type == um2::VTKElemType::Quad);
  ASSERT(conn == um2::Vector<Int>({4, 5, 7, 8}));
  type = um2::VTKElemType::Invalid;
  conn.clear();

  soup3.getElement(3, type, conn);
  ASSERT(type == um2::VTKElemType::Triangle);
  ASSERT(conn == um2::Vector<Int>({5, 6, 7}));

  // Check elsets
  ASSERT(soup3.numElsets() == 3);
  soup3.getElset(0, name, ids, elset_data);
  ASSERT(name == "A");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.empty());
  ids.clear();
  elset_data.clear();

  soup3.getElset(1, name, ids, elset_data);
  ASSERT(name == "B");
  ASSERT(ids == um2::Vector<Int>({2, 3}));
  ASSERT(elset_data.empty());
  ids.clear();
  elset_data.clear();

  soup3.getElset(2, name, ids, elset_data);
  ASSERT(name == "C");
  ASSERT(ids == um2::Vector<Int>({1, 2, 3}));
  ASSERT(elset_data.size() == 3);
  ASSERT_NEAR(elset_data[0], 11, eps);
  ASSERT_NEAR(elset_data[1], 22, eps);
  ASSERT_NEAR(elset_data[2], 33, eps);
  ids.clear();
  elset_data.clear();
}

TEST_CASE(io_abaqus_tri_mesh)
{
  um2::String const filename = "./mesh_files/tri.inp";
  um2::PolytopeSoup mesh_ref;
  makeReferenceTriPolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 10); // Only missing data
}

TEST_CASE(io_abaqus_quad_mesh)
{
  um2::String const filename = "./mesh_files/quad.inp";
  um2::PolytopeSoup mesh_ref;
  makeReferenceQuadPolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 10); // Only missing data
}

TEST_CASE(io_abaqus_tri_quad_mesh)
{
  um2::String const filename = "./mesh_files/tri_quad.inp";
  um2::PolytopeSoup mesh_ref;
  makeReferenceTriQuadPolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 10); // Only missing data
}

TEST_CASE(io_abaqus_tri6_mesh)
{
  um2::String const filename = "./mesh_files/tri6.inp";
  um2::PolytopeSoup mesh_ref;
  makeReferenceTri6PolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 10); // Only missing data
}

TEST_CASE(io_abaqus_quad8_mesh)
{
  um2::String const filename = "./mesh_files/quad8.inp";
  um2::PolytopeSoup mesh_ref;
  makeReferenceQuad8PolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 10); // Only missing data
}

TEST_CASE(io_abaqus_tri6_quad8_mesh)
{
  um2::String const filename = "./mesh_files/tri6_quad8.inp";
  um2::PolytopeSoup mesh_ref;
  makeReferenceTri6Quad8PolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 10); // Only missing data
}

TEST_CASE(io_vtk_tri_mesh)
{
  um2::String const filename = "./mesh_files/tri.vtk";
  um2::PolytopeSoup mesh_ref;
  makeReferenceTriPolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 6); // Different elsets
  ASSERT(mesh.numElsets() == 1);
  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  mesh.getElset(0, name, ids, elset_data);
  ASSERT(name == "flux");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.size() == 2);
  ASSERT_NEAR(elset_data[0], 1, castIfNot<Float>(1e-6));
  ASSERT_NEAR(elset_data[1], 2, castIfNot<Float>(1e-6));
}

TEST_CASE(io_vtk_quad_mesh)
{
  um2::String const filename = "./mesh_files/quad.vtk";
  um2::PolytopeSoup mesh_ref;
  makeReferenceQuadPolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 6); // Different elsets
  ASSERT(mesh.numElsets() == 1);
  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  mesh.getElset(0, name, ids, elset_data);
  ASSERT(name == "flux");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.size() == 2);
  ASSERT_NEAR(elset_data[0], 1, castIfNot<Float>(1e-6));
  ASSERT_NEAR(elset_data[1], 2, castIfNot<Float>(1e-6));
}

TEST_CASE(io_vtk_tri_quad_mesh)
{
  um2::String const filename = "./mesh_files/tri_quad.vtk";
  um2::PolytopeSoup mesh_ref;
  makeReferenceTriQuadPolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 6); // Different elsets
  ASSERT(mesh.numElsets() == 1);
  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  mesh.getElset(0, name, ids, elset_data);
  ASSERT(name == "flux");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.size() == 2);
  ASSERT_NEAR(elset_data[0], 1, castIfNot<Float>(1e-6));
  ASSERT_NEAR(elset_data[1], 2, castIfNot<Float>(1e-6));
}

TEST_CASE(io_vtk_tri6_mesh)
{
  um2::String const filename = "./mesh_files/tri6.vtk";
  um2::PolytopeSoup mesh_ref;
  makeReferenceTri6PolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 6); // Different elsets
  ASSERT(mesh.numElsets() == 1);
  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  mesh.getElset(0, name, ids, elset_data);
  ASSERT(name == "flux");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.size() == 2);
  ASSERT_NEAR(elset_data[0], 1, castIfNot<Float>(1e-6));
  ASSERT_NEAR(elset_data[1], 2, castIfNot<Float>(1e-6));
}

TEST_CASE(io_vtk_quad8_mesh)
{
  um2::String const filename = "./mesh_files/quad8.vtk";
  um2::PolytopeSoup mesh_ref;
  makeReferenceQuad8PolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 6); // Different elsets
  ASSERT(mesh.numElsets() == 1);
  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  mesh.getElset(0, name, ids, elset_data);
  ASSERT(name == "flux");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.size() == 2);
  ASSERT_NEAR(elset_data[0], 1, castIfNot<Float>(1e-6));
  ASSERT_NEAR(elset_data[1], 2, castIfNot<Float>(1e-6));
}

TEST_CASE(io_vtk_tri6_quad8_mesh)
{
  um2::String const filename = "./mesh_files/tri6_quad8.vtk";
  um2::PolytopeSoup mesh_ref;
  makeReferenceTri6Quad8PolytopeSoup(mesh_ref);

  um2::PolytopeSoup const mesh(filename);

  ASSERT(mesh.compare(mesh_ref) == 6); // Different elsets
  ASSERT(mesh.numElsets() == 1);
  um2::String name;
  um2::Vector<Int> ids;
  um2::Vector<Float> elset_data;
  mesh.getElset(0, name, ids, elset_data);
  ASSERT(name == "flux");
  ASSERT(ids == um2::Vector<Int>({0, 1}));
  ASSERT(elset_data.size() == 2);
  ASSERT_NEAR(elset_data[0], 1, castIfNot<Float>(1e-6));
  ASSERT_NEAR(elset_data[1], 2, castIfNot<Float>(1e-6));
}

#if UM2_HAS_XDMF
TEST_CASE(io_xdmf_tri_mesh)
{
  um2::PolytopeSoup mesh_ref;
  makeReferenceTriPolytopeSoup(mesh_ref);
  mesh_ref.write("./tri.xdmf");

  um2::PolytopeSoup mesh;
  mesh.read("./tri.xdmf");

  ASSERT(mesh.compare(mesh_ref) == 0);

  int stat = std::remove("./tri.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./tri.h5");
  ASSERT(stat == 0);
}

TEST_CASE(io_xdmf_quad_mesh)
{
  um2::PolytopeSoup mesh_ref;
  makeReferenceQuadPolytopeSoup(mesh_ref);
  mesh_ref.write("./quad.xdmf");

  um2::PolytopeSoup mesh;
  mesh.read("./quad.xdmf");

  ASSERT(mesh.compare(mesh_ref) == 0);

  int stat = std::remove("./quad.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./quad.h5");
  ASSERT(stat == 0);
}

TEST_CASE(io_xdmf_tri_quad_mesh)
{
  um2::PolytopeSoup mesh_ref;
  makeReferenceTriQuadPolytopeSoup(mesh_ref);
  mesh_ref.write("./tri_quad.xdmf");

  um2::PolytopeSoup mesh;
  mesh.read("./tri_quad.xdmf");

  ASSERT(mesh.compare(mesh_ref) == 0);

  int stat = std::remove("./tri_quad.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./tri_quad.h5");
  ASSERT(stat == 0);
}

TEST_CASE(io_xdmf_tri6_mesh)
{
  um2::PolytopeSoup mesh_ref;
  makeReferenceTri6PolytopeSoup(mesh_ref);
  mesh_ref.write("./tri6.xdmf");

  um2::PolytopeSoup mesh;
  mesh.read("./tri6.xdmf");

  ASSERT(mesh.compare(mesh_ref) == 0);

  int stat = std::remove("./tri6.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./tri6.h5");
  ASSERT(stat == 0);
}

TEST_CASE(io_xdmf_quad8_mesh)
{
  um2::PolytopeSoup mesh_ref;
  makeReferenceQuad8PolytopeSoup(mesh_ref);
  mesh_ref.write("./quad8.xdmf");

  um2::PolytopeSoup mesh;
  mesh.read("./quad8.xdmf");

  ASSERT(mesh.compare(mesh_ref) == 0);

  int stat = std::remove("./quad8.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./quad8.h5");
  ASSERT(stat == 0);
}

TEST_CASE(io_xdmf_tri6_quad8_mesh)
{
  um2::PolytopeSoup mesh_ref;
  makeReferenceTri6Quad8PolytopeSoup(mesh_ref);
  mesh_ref.write("./tri6_quad8.xdmf");

  um2::PolytopeSoup mesh;
  mesh.read("./tri6_quad8.xdmf");

  ASSERT(mesh.compare(mesh_ref) == 0);

  int stat = std::remove("./tri6_quad8.xdmf");
  ASSERT(stat == 0);
  stat = std::remove("./tri6_quad8.h5");
  ASSERT(stat == 0);
}
#endif // UM2_HAS_XDMF

// TEST_CASE(getPowerRegions)
//{
//   //      Face ID                Power
//   // -----------------    -----------------
//   // | 0 | 1 | 2 | 3 |    | 2 | 1 | 0 | 0 |
//   // -----------------    -----------------
//   // | 4 | 5 | 6 | 7 |    | 1 | 0 | 0 | 0 |
//   // ----------------- -> -----------------
//   // | 8 | 9 |10 |11 |    | 0 | 0 | 1 | 1 |
//   // -----------------    -----------------
//   // |12 |13 |14 |15 |    | 4 | 0 | 1 | 1 |
//   // -----------------    -----------------
//
//   // This should yield 3 regions with
//   // Power    |   Centroid
//   // ---------------------
//   // 4        | (1/2, 1/2)
//   // 3        | (11/4, 1)
//   // 4        | (5/6, 19/6)
//
//   um2::PolytopeSoup mesh;
//   for (Int j = 0; j < 5; ++j) {
//     for (Int i = 0; i < 5; ++i) {
//       if (i == 4) {
//         auto const x = castIfNot<Float>(3.5);
//         auto const y = castIfNot<Float>(j);
//         mesh.addVertex(x, y);
//       } else {
//         mesh.addVertex(um2::Point3F(i, j, 0));
//       }
//     }
//   }
//
//   um2::Vector<Int> conn(4);
//   um2::VTKElemType const type = um2::VTKElemType::Quad;
//   for (Int j = 0; j < 4; ++j) {
//     for (Int i = 0; i < 4; ++i) {
//       conn[0] = i + j * 5;
//       conn[1] = i + 1 + j * 5;
//       conn[2] = i + 1 + (j + 1) * 5;
//       conn[3] = i + (j + 1) * 5;
//       mesh.addElement(type, conn);
//     }
//   }
//
//   um2::Vector<Float> powers(16);
//   powers[0] = 4;
//   powers[2] = 1;
//   powers[3] = 1;
//   powers[6] = 1;
//   powers[7] = 1;
//   powers[8] = 1;
//   powers[12] = 2;
//   powers[13] = 1;
//   um2::Vector<Int> faces(16);
//   um2::iota(faces.begin(), faces.end(), 0);
//   mesh.addElset("power", faces, powers);
//
//   auto const subset_pc = um2::getPowerRegions(mesh);
//   // Print the results
//   ASSERT(subset_pc.size() == 3);
//   auto const eps = castIfNot<Float>(1e-6);
//
//   ASSERT_NEAR(subset_pc[0].first, 4, eps);
//   ASSERT_NEAR(subset_pc[0].second[0], castIfNot<Float>(1) / castIfNot<Float>(2), eps);
//   ASSERT_NEAR(subset_pc[0].second[1], castIfNot<Float>(1) / castIfNot<Float>(2), eps);
//
//   ASSERT_NEAR(subset_pc[1].first, 3, eps);
//   ASSERT_NEAR(subset_pc[1].second[0], castIfNot<Float>(11) / castIfNot<Float>(4), eps);
//   ASSERT_NEAR(subset_pc[1].second[1], 1, eps);
//
//   ASSERT_NEAR(subset_pc[2].first, 4, eps);
//   ASSERT_NEAR(subset_pc[2].second[0], castIfNot<Float>(5) / castIfNot<Float>(6), eps);
//   ASSERT_NEAR(subset_pc[2].second[1], castIfNot<Float>(19) / castIfNot<Float>(6), eps);
// }

TEST_SUITE(PolytopeSoup)
{
  TEST(addVertex);
  TEST(addElement);
  TEST(addElset);
  TEST(sortElsets);
  TEST(getSubset);
  TEST(operator_plus_equal);
  TEST(io_abaqus_tri_mesh);
  TEST(io_abaqus_quad_mesh);
  TEST(io_abaqus_tri_quad_mesh);
  TEST(io_abaqus_tri6_mesh);
  TEST(io_abaqus_quad8_mesh);
  TEST(io_abaqus_tri6_quad8_mesh);
  TEST(io_vtk_tri_mesh);
  TEST(io_vtk_quad_mesh);
  TEST(io_vtk_tri_quad_mesh);
  TEST(io_vtk_tri6_mesh);
  TEST(io_vtk_quad8_mesh);
  TEST(io_vtk_tri6_quad8_mesh);
#if UM2_HAS_XDMF
  TEST(io_xdmf_tri_mesh);
  TEST(io_xdmf_quad_mesh);
  TEST(io_xdmf_tri_quad_mesh);
  TEST(io_xdmf_tri6_mesh);
  TEST(io_xdmf_quad8_mesh);
  TEST(io_xdmf_tri6_quad8_mesh);
#endif
  //  TEST(getPowerRegions);
}

auto
main() -> int
{
  RUN_SUITE(PolytopeSoup);
  return 0;
}
