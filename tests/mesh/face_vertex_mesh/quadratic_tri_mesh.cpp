#include <um2/mesh/face_vertex_mesh.hpp>

#include "../helpers/setup_mesh.hpp"
#include "../helpers/setup_polytope_soup.hpp"

#include "../../test_macros.hpp"

F constexpr eps = condCast<F>(1e-6);

HOSTDEV
TEST_CASE(accessors)
{
  um2::Tri6FVM const mesh = makeTri6ReferenceMesh();
  ASSERT(mesh.numVertices() == 9);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::QuadraticTriangle2 tri0_ref(mesh.getVertex(0), mesh.getVertex(1),
                                   mesh.getVertex(2), mesh.getVertex(3),
                                   mesh.getVertex(4), mesh.getVertex(5));
  auto const tri0 = mesh.getFace(0);
  ASSERT(um2::isApprox(tri0[0], tri0_ref[0]));
  ASSERT(um2::isApprox(tri0[1], tri0_ref[1]));
  ASSERT(um2::isApprox(tri0[2], tri0_ref[2]));
  ASSERT(um2::isApprox(tri0[3], tri0_ref[3]));
  ASSERT(um2::isApprox(tri0[4], tri0_ref[4]));
  ASSERT(um2::isApprox(tri0[5], tri0_ref[5]));
  um2::QuadraticTriangle2 tri1_ref(mesh.getVertex(1), mesh.getVertex(6),
                                   mesh.getVertex(2), mesh.getVertex(7),
                                   mesh.getVertex(8), mesh.getVertex(4));
  auto const tri1 = mesh.getFace(1);
  ASSERT(um2::isApprox(tri1[0], tri1_ref[0]));
  ASSERT(um2::isApprox(tri1[1], tri1_ref[1]));
  ASSERT(um2::isApprox(tri1[2], tri1_ref[2]));
  ASSERT(um2::isApprox(tri1[3], tri1_ref[3]));
  ASSERT(um2::isApprox(tri1[4], tri1_ref[4]));
  ASSERT(um2::isApprox(tri1[5], tri1_ref[5]));
}

TEST_CASE(addVertex_addFace)
{
  um2::Tri6FVM mesh;
  mesh.addVertex({0, 0});
  mesh.addVertex({1, 0});
  mesh.addVertex({0, 1});
  mesh.addVertex({condCast<F>(0.5), condCast<F>(0.0)});
  mesh.addVertex({condCast<F>(0.7), condCast<F>(0.5)});
  mesh.addVertex({condCast<F>(0.0), condCast<F>(0.5)});
  mesh.addVertex({1, 1});
  mesh.addVertex({condCast<F>(1.0), condCast<F>(0.5)});
  mesh.addVertex({condCast<F>(0.5), condCast<F>(1.0)});
  mesh.addFace({0, 1, 2, 3, 4, 5});
  mesh.addFace({1, 6, 2, 7, 8, 4});
  um2::QuadraticTriangle2 tri0_ref(mesh.getVertex(0), mesh.getVertex(1),
                                   mesh.getVertex(2), mesh.getVertex(3),
                                   mesh.getVertex(4), mesh.getVertex(5));
  auto const tri0 = mesh.getFace(0);
  ASSERT(um2::isApprox(tri0[0], tri0_ref[0]));
  ASSERT(um2::isApprox(tri0[1], tri0_ref[1]));
  ASSERT(um2::isApprox(tri0[2], tri0_ref[2]));
  ASSERT(um2::isApprox(tri0[3], tri0_ref[3]));
  ASSERT(um2::isApprox(tri0[4], tri0_ref[4]));
  ASSERT(um2::isApprox(tri0[5], tri0_ref[5]));
  um2::QuadraticTriangle2 tri1_ref(mesh.getVertex(1), mesh.getVertex(6),
                                   mesh.getVertex(2), mesh.getVertex(7),
                                   mesh.getVertex(8), mesh.getVertex(4));
  auto const tri1 = mesh.getFace(1);
  ASSERT(um2::isApprox(tri1[0], tri1_ref[0]));
  ASSERT(um2::isApprox(tri1[1], tri1_ref[1]));
  ASSERT(um2::isApprox(tri1[2], tri1_ref[2]));
  ASSERT(um2::isApprox(tri1[3], tri1_ref[3]));
  ASSERT(um2::isApprox(tri1[4], tri1_ref[4]));
  ASSERT(um2::isApprox(tri1[5], tri1_ref[5]));
}

TEST_CASE(poly_soup_constructor)
{
  um2::PolytopeSoup poly_soup;
  makeReferenceTri6PolytopeSoup(poly_soup);
  um2::Tri6FVM const mesh_ref = makeTri6ReferenceMesh();
  um2::Tri6FVM const mesh(poly_soup);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (I i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(um2::isApprox(mesh.getVertex(i), mesh_ref.getVertex(i)));
  }
  for (I i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    auto const face_ref = mesh_ref.getFace(i);
    for (I j = 0; j < 6; ++j) {
      ASSERT(um2::isApprox(face[j], face_ref[j]));
    }
  }
}

TEST_CASE(boundingBox)
{
  um2::Tri6FVM const mesh = makeTri6ReferenceMesh();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), condCast<F>(0), eps);
  ASSERT_NEAR(box.xMax(), condCast<F>(1), eps);
  ASSERT_NEAR(box.yMin(), condCast<F>(0), eps);
  ASSERT_NEAR(box.yMax(), condCast<F>(1), eps);
}

TEST_CASE(faceContaining)
{
  um2::Tri6FVM const mesh = makeTri6ReferenceMesh();
  um2::Point2 p(condCast<F>(0.6), condCast<F>(0.5));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2(condCast<F>(0.8), condCast<F>(0.5));
  ASSERT(mesh.faceContaining(p) == 1);
}

TEST_CASE(populateVF)
{
  um2::Tri6FVM mesh = makeTri6ReferenceMesh();
  ASSERT(mesh.vertexFaceOffsets().empty());
  ASSERT(mesh.vertexFaceConn().empty());
  mesh.populateVF();
  um2::Vector<I> const vf_offsets_ref = {0, 1, 3, 5, 6, 8, 9, 10, 11, 12};
  um2::Vector<I> const vf_ref = {0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1};
  ASSERT(mesh.vertexFaceOffsets() == vf_offsets_ref);
  ASSERT(mesh.vertexFaceConn() == vf_ref);
}

TEST_CASE(intersect)
{
  um2::Tri6FVM const mesh = makeTri6ReferenceMesh();
  um2::Point2 const origin(0, 0);
  um2::Vec2<F> direction(condCast<F>(0.7), condCast<F>(0.5));
  direction.normalize();
  um2::Ray2 const ray(origin, direction);
  um2::Vector<F> intersections;
  mesh.intersect(ray, intersections);
  ASSERT(intersections.size() == 5);
  F const it1 = um2::sqrt(condCast<F>(0.74));
  F const it2 = 1 / (condCast<F>(0.7) / it1);
  ASSERT_NEAR(intersections[0], 0, eps);
  ASSERT_NEAR(intersections[1], 0, eps);
  ASSERT_NEAR(intersections[2], it1, eps);
  ASSERT_NEAR(intersections[3], it1, eps);
  ASSERT_NEAR(intersections[4], it2, eps);
}

// TEST_CASE(toPolytopeSoup)
//{
//   um2::Tri6FVM const quad_mesh = makeTri6ReferenceMesh();
//   um2::PolytopeSoup quad_poly_soup_ref;
//   makeReferenceTri6PolytopeSoup(quad_poly_soup_ref);
//   um2::PolytopeSoup quad_poly_soup;
//   quad_mesh.toPolytopeSoup(quad_poly_soup);
//   ASSERT(quad_poly_soup.compareTo(quad_poly_soup_ref) == 10);
//   ASSERT(quad_poly_soup.getMeshType() == um2::MeshType::QuadraticTri);
// }

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(accessors)
#endif

TEST_SUITE(Tri6FVM)
{
  TEST_HOSTDEV(accessors);
  TEST(addVertex_addFace);
  TEST(poly_soup_constructor);
  TEST(boundingBox);
  TEST(faceContaining);
  TEST(populateVF);
  TEST(intersect);

  //  TEST((toPolytopeSoup));
}

auto
main() -> int
{
  RUN_SUITE(Tri6FVM);
  return 0;
}
