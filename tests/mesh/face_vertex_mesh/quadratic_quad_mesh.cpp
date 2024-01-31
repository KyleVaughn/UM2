#include <um2/mesh/face_vertex_mesh.hpp>

#include "../helpers/setup_mesh.hpp"
#include "../helpers/setup_polytope_soup.hpp"

#include "../../test_macros.hpp"

F constexpr eps = condCast<F>(1e-6);

HOSTDEV
TEST_CASE(accessors)
{
  um2::Quad8FVM const mesh = makeQuad8ReferenceMesh();
  ASSERT(mesh.numVertices() == 13);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::QuadraticQuadrilateral2 quad0_ref(
      mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(2), mesh.getVertex(3),
      mesh.getVertex(6), mesh.getVertex(7), mesh.getVertex(8), mesh.getVertex(9));
  auto const quad0 = mesh.getFace(0);
  ASSERT(um2::isApprox(quad0[0], quad0_ref[0]));
  ASSERT(um2::isApprox(quad0[1], quad0_ref[1]));
  ASSERT(um2::isApprox(quad0[2], quad0_ref[2]));
  ASSERT(um2::isApprox(quad0[3], quad0_ref[3]));
  ASSERT(um2::isApprox(quad0[4], quad0_ref[4]));
  ASSERT(um2::isApprox(quad0[5], quad0_ref[5]));
  ASSERT(um2::isApprox(quad0[6], quad0_ref[6]));
  ASSERT(um2::isApprox(quad0[7], quad0_ref[7]));
  um2::QuadraticQuadrilateral2 quad1_ref(
      mesh.getVertex(1), mesh.getVertex(4), mesh.getVertex(5), mesh.getVertex(2),
      mesh.getVertex(10), mesh.getVertex(11), mesh.getVertex(12), mesh.getVertex(7));
  auto const quad1 = mesh.getFace(1);
  ASSERT(um2::isApprox(quad1[0], quad1_ref[0]));
  ASSERT(um2::isApprox(quad1[1], quad1_ref[1]));
  ASSERT(um2::isApprox(quad1[2], quad1_ref[2]));
  ASSERT(um2::isApprox(quad1[3], quad1_ref[3]));
  ASSERT(um2::isApprox(quad1[4], quad1_ref[4]));
  ASSERT(um2::isApprox(quad1[5], quad1_ref[5]));
  ASSERT(um2::isApprox(quad1[6], quad1_ref[6]));
  ASSERT(um2::isApprox(quad1[7], quad1_ref[7]));
}

HOSTDEV
TEST_CASE(addVertex_addFace)
{
  um2::Quad8FVM mesh;
  mesh.addVertex({condCast<F>(0.0), condCast<F>(0.0)});
  mesh.addVertex({condCast<F>(1.0), condCast<F>(0.0)});
  mesh.addVertex({condCast<F>(1.0), condCast<F>(1.0)});
  mesh.addVertex({condCast<F>(0.0), condCast<F>(1.0)});
  mesh.addVertex({condCast<F>(2.0), condCast<F>(0.0)});
  mesh.addVertex({condCast<F>(2.0), condCast<F>(1.0)});
  mesh.addVertex({condCast<F>(0.5), condCast<F>(0.0)});
  mesh.addVertex({condCast<F>(1.1), condCast<F>(0.6)});
  mesh.addVertex({condCast<F>(0.5), condCast<F>(1.0)});
  mesh.addVertex({condCast<F>(0.0), condCast<F>(0.5)});
  mesh.addVertex({condCast<F>(1.5), condCast<F>(0.0)});
  mesh.addVertex({condCast<F>(2.0), condCast<F>(0.5)});
  mesh.addVertex({condCast<F>(1.5), condCast<F>(1.0)});
  mesh.addFace({0, 1, 2, 3, 6, 7, 8, 9});
  mesh.addFace({1, 4, 5, 2, 10, 11, 12, 7});
  // face
  um2::QuadraticQuadrilateral2 quad0_ref(
      mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(2), mesh.getVertex(3),
      mesh.getVertex(6), mesh.getVertex(7), mesh.getVertex(8), mesh.getVertex(9));
  auto const quad0 = mesh.getFace(0);
  ASSERT(um2::isApprox(quad0[0], quad0_ref[0]));
  ASSERT(um2::isApprox(quad0[1], quad0_ref[1]));
  ASSERT(um2::isApprox(quad0[2], quad0_ref[2]));
  ASSERT(um2::isApprox(quad0[3], quad0_ref[3]));
  ASSERT(um2::isApprox(quad0[4], quad0_ref[4]));
  ASSERT(um2::isApprox(quad0[5], quad0_ref[5]));
  ASSERT(um2::isApprox(quad0[6], quad0_ref[6]));
  ASSERT(um2::isApprox(quad0[7], quad0_ref[7]));
  um2::QuadraticQuadrilateral2 quad1_ref(
      mesh.getVertex(1), mesh.getVertex(4), mesh.getVertex(5), mesh.getVertex(2),
      mesh.getVertex(10), mesh.getVertex(11), mesh.getVertex(12), mesh.getVertex(7));
  auto const quad1 = mesh.getFace(1);
  ASSERT(um2::isApprox(quad1[0], quad1_ref[0]));
  ASSERT(um2::isApprox(quad1[1], quad1_ref[1]));
  ASSERT(um2::isApprox(quad1[2], quad1_ref[2]));
  ASSERT(um2::isApprox(quad1[3], quad1_ref[3]));
  ASSERT(um2::isApprox(quad1[4], quad1_ref[4]));
  ASSERT(um2::isApprox(quad1[5], quad1_ref[5]));
  ASSERT(um2::isApprox(quad1[6], quad1_ref[6]));
  ASSERT(um2::isApprox(quad1[7], quad1_ref[7]));
}

TEST_CASE(poly_soup_constructor)
{
  um2::PolytopeSoup poly_soup;
  makeReferenceQuad8PolytopeSoup(poly_soup);
  um2::Quad8FVM const mesh_ref = makeQuad8ReferenceMesh();
  um2::Quad8FVM const mesh(poly_soup);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (I i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(um2::isApprox(mesh.getVertex(i), mesh_ref.getVertex(i)));
  }
  for (I i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    auto const face_ref = mesh_ref.getFace(i);
    for (I j = 0; j < 8; ++j) {
      ASSERT(um2::isApprox(face[j], face_ref[j]));
    }
  }
}

TEST_CASE(boundingBox)
{
  um2::Quad8FVM const mesh = makeQuad8ReferenceMesh();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), condCast<F>(0), eps);
  ASSERT_NEAR(box.xMax(), condCast<F>(2), eps);
  ASSERT_NEAR(box.yMin(), condCast<F>(0), eps);
  ASSERT_NEAR(box.yMax(), condCast<F>(1), eps);
}

TEST_CASE(faceContaining)
{
  um2::Quad8FVM const mesh = makeQuad8ReferenceMesh();
  um2::Point2 p(condCast<F>(1.05), condCast<F>(0.6));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2(condCast<F>(1.15), condCast<F>(0.6));
  ASSERT(mesh.faceContaining(p) == 1);
}

TEST_CASE(populateVF)
{
  um2::Quad8FVM mesh = makeQuad8ReferenceMesh();
  ASSERT(mesh.vertexFaceOffsets().empty());
  ASSERT(mesh.vertexFaceConn().empty());
  mesh.populateVF();
  um2::Vector<I> const vf_offsets_ref = {0, 1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16};
  um2::Vector<I> const vf_ref = {0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1};
  ASSERT(mesh.vertexFaceOffsets() == vf_offsets_ref);
  ASSERT(mesh.vertexFaceConn() == vf_ref);
}

TEST_CASE(intersect)
{
  um2::Quad8FVM const mesh = makeQuad8ReferenceMesh();
  um2::Point2 const origin(0, 0);
  um2::Vec2<F> direction(condCast<F>(1.1), condCast<F>(0.6));
  direction.normalize();
  um2::Ray2 const ray(origin, direction);
  um2::Vector<F> intersections;
  mesh.intersect(ray, intersections);
  ASSERT(intersections.size() == 5);
  F const int1 = um2::sqrt(condCast<F>(1.57));
  F const int2 = 1 / (condCast<F>(0.6) / int1);
  ASSERT_NEAR(intersections[0], 0, eps);
  ASSERT_NEAR(intersections[1], 0, eps);
  ASSERT_NEAR(intersections[2], int1, eps);
  ASSERT_NEAR(intersections[3], int1, eps);
  ASSERT_NEAR(intersections[4], int2, eps);
}

TEST_CASE(mortonSort)
{
  // 16 --- 17 --- 18 --- 19 --- 20
  //  |             |             |
  //  |             |             |
  // 13      F2    14     F1     15
  //  |             |             |
  //  |             |             |
  //  8 ---  9 --- 10 --- 11 --- 12
  //  |             |             |
  //  |             |             |
  //  5      F3     6     F0      7
  //  |             |             |
  //  |             |             |
  //  0 ---  1 ---  2 ---  3 ---  4
  //
  um2::Quad8FVM mesh;
  mesh.addVertex({0, 0});
  mesh.addVertex({1, 0});
  mesh.addVertex({2, 0});
  mesh.addVertex({3, 0});
  mesh.addVertex({4, 0});

  mesh.addVertex({0, 1});
  mesh.addVertex({2, 1});
  mesh.addVertex({4, 1});

  mesh.addVertex({0, 2});
  mesh.addVertex({1, 2});
  mesh.addVertex({2, 2});
  mesh.addVertex({3, 2});
  mesh.addVertex({4, 2});

  mesh.addVertex({0, 3});
  mesh.addVertex({2, 3});
  mesh.addVertex({4, 3});

  mesh.addVertex({0, 4});
  mesh.addVertex({1, 4});
  mesh.addVertex({2, 4});
  mesh.addVertex({3, 4});
  mesh.addVertex({4, 4});

  mesh.addFace({2, 4, 12, 10, 3, 7, 11, 6});
  mesh.addFace({10, 12, 20, 18, 11, 15, 19, 14});
  mesh.addFace({8, 10, 18, 16, 9, 14, 17, 13});
  mesh.addFace({0, 2, 10, 8, 1, 6, 9, 5});

  mesh.mortonSort();

  auto const f0 = mesh.getFace(0);
  auto const f1 = mesh.getFace(1);
  auto const f2 = mesh.getFace(2);
  auto const f3 = mesh.getFace(3);
  ASSERT(um2::isApprox(f0.centroid(), um2::Point2(1, 1)));
  ASSERT(um2::isApprox(f1.centroid(), um2::Point2(3, 1)));
  ASSERT(um2::isApprox(f2.centroid(), um2::Point2(1, 3)));
  ASSERT(um2::isApprox(f3.centroid(), um2::Point2(3, 3)));

  ASSERT(um2::isApprox(mesh.getVertex(0), um2::Point2(0, 0)));
  ASSERT(um2::isApprox(mesh.getVertex(1), um2::Point2(1, 0)));
  ASSERT(um2::isApprox(mesh.getVertex(2), um2::Point2(0, 1)));
  ASSERT(um2::isApprox(mesh.getVertex(7), um2::Point2(2, 2)));
  ASSERT(um2::isApprox(mesh.getVertex(20), um2::Point2(4, 4)));
}

// TEST_CASE(toPolytopeSoup)
//{
//   um2::Quad8FVM const quad_mesh = makeQuadReferenceMesh();
//   um2::PolytopeSoup quad_poly_soup_ref;
//   makeReferenceQuad8PolytopeSoup(quad_poly_soup_ref);
//   um2::PolytopeSoup quad_poly_soup;
//   quad_mesh.toPolytopeSoup(quad_poly_soup);
//   ASSERT(quad_poly_soup.comapreTo(quad_poly_soup_ref) == 10);
//   ASSERT(quad_poly_soup.getMeshType() == um2::MeshType::QuadraticQuad);
// }

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(accessors)
#endif

TEST_SUITE(Quad8FVM)
{
  TEST_HOSTDEV(accessors);
  TEST((addVertex_addFace));
  TEST((poly_soup_constructor));
  TEST((boundingBox));
  TEST((faceContaining));
  TEST((populateVF));
  TEST((intersect));
  TEST((mortonSort));
}

auto
main() -> int
{
  RUN_SUITE(Quad8FVM);
  return 0;
}
