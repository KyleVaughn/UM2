#include <um2/mesh/face_vertex_mesh.hpp>

#include "../helpers/setup_mesh.hpp"
#include "../helpers/setup_polytope_soup.hpp"

#include "../../test_macros.hpp"

F constexpr eps = condCast<F>(1e-6);

HOSTDEV
TEST_CASE(accessors)
{
  um2::QuadFVM const mesh = makeQuadReferenceMesh();
  ASSERT(mesh.numVertices() == 6);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::Quadrilateral2 quad0_ref(mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(2),
                                mesh.getVertex(3));
  auto const quad0 = mesh.getFace(0);
  ASSERT(um2::isApprox(quad0[0], quad0_ref[0]));
  ASSERT(um2::isApprox(quad0[1], quad0_ref[1]));
  ASSERT(um2::isApprox(quad0[2], quad0_ref[2]));
  ASSERT(um2::isApprox(quad0[3], quad0_ref[3]));
  um2::Quadrilateral2 quad1_ref(mesh.getVertex(1), mesh.getVertex(4), mesh.getVertex(5),
                                mesh.getVertex(2));
  auto const quad1 = mesh.getFace(1);
  ASSERT(um2::isApprox(quad1[0], quad1_ref[0]));
  ASSERT(um2::isApprox(quad1[1], quad1_ref[1]));
  ASSERT(um2::isApprox(quad1[2], quad1_ref[2]));
  ASSERT(um2::isApprox(quad1[3], quad1_ref[3]));
}

TEST_CASE(addVertex_addFace)
{
  um2::QuadFVM mesh;
  mesh.addVertex({0, 0});
  mesh.addVertex({1, 0});
  mesh.addVertex({1, 1});
  mesh.addVertex({0, 1});
  mesh.addVertex({2, 0});
  mesh.addVertex({2, 1});
  mesh.addFace({0, 1, 2, 3});
  mesh.addFace({1, 4, 5, 2});
  // Same as reference mesh. Should make an == operator for meshes.
  um2::Quadrilateral2 quad0_ref(mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(2),
                                mesh.getVertex(3));
  auto const quad0 = mesh.getFace(0);
  ASSERT(um2::isApprox(quad0[0], quad0_ref[0]));
  ASSERT(um2::isApprox(quad0[1], quad0_ref[1]));
  ASSERT(um2::isApprox(quad0[2], quad0_ref[2]));
  ASSERT(um2::isApprox(quad0[3], quad0_ref[3]));
  um2::Quadrilateral2 quad1_ref(mesh.getVertex(1), mesh.getVertex(4), mesh.getVertex(5),
                                mesh.getVertex(2));
  auto const quad1 = mesh.getFace(1);
  ASSERT(um2::isApprox(quad1[0], quad1_ref[0]));
  ASSERT(um2::isApprox(quad1[1], quad1_ref[1]));
  ASSERT(um2::isApprox(quad1[2], quad1_ref[2]));
  ASSERT(um2::isApprox(quad1[3], quad1_ref[3]));
}

TEST_CASE(poly_soup_constructor)
{
  um2::PolytopeSoup poly_soup;
  makeReferenceQuadPolytopeSoup(poly_soup);
  um2::QuadFVM const mesh_ref = makeQuadReferenceMesh();
  um2::QuadFVM const mesh(poly_soup);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (I i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(um2::isApprox(mesh.getVertex(i), mesh_ref.getVertex(i)));
  }
  for (I i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    auto const face_ref = mesh_ref.getFace(i);
    for (I j = 0; j < 4; ++j) {
      ASSERT(um2::isApprox(face[j], face_ref[j]));
    }
  }
}

TEST_CASE(boundingBox)
{
  um2::QuadFVM const mesh = makeQuadReferenceMesh();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), condCast<F>(0), eps);
  ASSERT_NEAR(box.xMax(), condCast<F>(2), eps);
  ASSERT_NEAR(box.yMin(), condCast<F>(0), eps);
  ASSERT_NEAR(box.yMax(), condCast<F>(1), eps);
}

TEST_CASE(faceContaining)
{
  um2::QuadFVM const mesh = makeQuadReferenceMesh();
  um2::Point2 p(condCast<F>(0.5), condCast<F>(0.25));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2(condCast<F>(1.5), condCast<F>(0.75));
  ASSERT(mesh.faceContaining(p) == 1);
}

TEST_CASE(populateVF)
{
  um2::QuadFVM mesh = makeQuadReferenceMesh();
  ASSERT(mesh.vertexFaceOffsets().empty());
  ASSERT(mesh.vertexFaceConn().empty());
  mesh.populateVF();
  um2::Vector<I> const vf_offsets_ref = {0, 1, 3, 5, 6, 7, 8};
  um2::Vector<I> const vf_ref = {0, 0, 1, 0, 1, 0, 1, 1};
  ASSERT(mesh.vertexFaceOffsets() == vf_offsets_ref);
  ASSERT(mesh.vertexFaceConn() == vf_ref);
}

TEST_CASE(intersect)
{
  um2::QuadFVM const mesh = makeQuadReferenceMesh();
  um2::Point2 const origin(condCast<F>(0), condCast<F>(-0.5));
  um2::Vec2<F> direction(1, 1);
  direction.normalize();
  um2::Ray2 const ray(origin, direction);
  um2::Vector<F> intersections;
  mesh.intersect(ray, intersections);
  ASSERT(intersections.size() == 4);
  F const sqrt_half = um2::sqrt(condCast<F>(0.5));
  ASSERT_NEAR(intersections[0], sqrt_half, eps);
  ASSERT_NEAR(intersections[1], 2 * sqrt_half, eps);
  ASSERT_NEAR(intersections[2], 2 * sqrt_half, eps);
  ASSERT_NEAR(intersections[3], 3 * sqrt_half, eps);
}

// TEST_CASE(toPolytopeSoup)
//{
//   um2::QuadFVM const quad_mesh = makeQuadReferenceMesh();
//   um2::PolytopeSoup quad_poly_soup_ref;
//   makeReferenceQuadPolytopeSoup(quad_poly_soup_ref);
//   um2::PolytopeSoup quad_poly_soup;
//   quad_mesh.toPolytopeSoup(quad_poly_soup);
//   ASSERT(quad_poly_soup.compareTo(quad_poly_soup_ref) == 10);
//   ASSERT(quad_poly_soup.getMeshType() == um2::MeshType::Quad);
// }

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(accessors);
#endif

TEST_SUITE(QuadFVM)
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
  RUN_SUITE(QuadFVM);
  return 0;
}
