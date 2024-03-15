#include <um2/mesh/face_vertex_mesh.hpp>

#include "../helpers/setup_mesh.hpp"
#include "../helpers/setup_polytope_soup.hpp"

#include "../../test_macros.hpp"

auto constexpr eps = castIfNot<Float>(1e-6);

HOSTDEV
TEST_CASE(accessors)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();
  ASSERT(mesh.numVertices() == 4);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::Triangle<2> tri0_ref(mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(2));
  auto const tri0 = mesh.getFace(0);
  ASSERT(tri0[0].isApprox(tri0_ref[0]));
  ASSERT(tri0[1].isApprox(tri0_ref[1]));
  ASSERT(tri0[2].isApprox(tri0_ref[2]));
  um2::Triangle<2> tri1_ref(mesh.getVertex(2), mesh.getVertex(3), mesh.getVertex(0));
  auto const tri1 = mesh.getFace(1);
  ASSERT(tri1[0].isApprox(tri1_ref[0]));
  ASSERT(tri1[1].isApprox(tri1_ref[1]));
  ASSERT(tri1[2].isApprox(tri1_ref[2]));
}

TEST_CASE(addVertex_addFace)
{
  um2::TriFVM mesh;
  mesh.addVertex({0, 0});
  mesh.addVertex({1, 0});
  mesh.addVertex({1, 1});
  mesh.addVertex({0, 1});
  mesh.addFace({0, 1, 2});
  mesh.addFace({2, 3, 0});
  // Same as reference mesh. Should make an == operator for meshes.
  um2::Triangle<2> tri0_ref(mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(2));
  auto const tri0 = mesh.getFace(0);
  ASSERT(tri0[0].isApprox(tri0_ref[0]));
  ASSERT(tri0[1].isApprox(tri0_ref[1]));
  ASSERT(tri0[2].isApprox(tri0_ref[2]));
  um2::Triangle<2> tri1_ref(mesh.getVertex(2), mesh.getVertex(3), mesh.getVertex(0));
  auto const tri1 = mesh.getFace(1);
  ASSERT(tri1[0].isApprox(tri1_ref[0]));
  ASSERT(tri1[1].isApprox(tri1_ref[1]));
  ASSERT(tri1[2].isApprox(tri1_ref[2]));
}

//TEST_CASE(poly_soup_constructor)
//{
//  um2::PolytopeSoup poly_soup;
//  makeReferenceTriPolytopeSoup(poly_soup);
//  um2::TriFVM const mesh_ref = makeTriReferenceMesh();
//  um2::TriFVM const mesh(poly_soup);
//  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
//  for (I i = 0; i < mesh.numVertices(); ++i) {
//    ASSERT(um2::isApprox(mesh.getVertex(i), mesh_ref.getVertex(i)));
//  }
//  for (I i = 0; i < mesh.numFaces(); ++i) {
//    auto const face = mesh.getFace(i);
//    auto const face_ref = mesh_ref.getFace(i);
//    for (I j = 0; j < 3; ++j) {
//      ASSERT(um2::isApprox(face[j], face_ref[j]));
//    }
//  }
//}

TEST_CASE(boundingBox)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.xMax(), castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.yMin(), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.yMax(), castIfNot<Float>(1), eps);
}

TEST_CASE(faceContaining)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();
  um2::Point2 p(castIfNot<Float>(0.5), castIfNot<Float>(0.25));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.75));
  ASSERT(mesh.faceContaining(p) == 1);
}

//TEST_CASE(populateVF)
//{
//  um2::TriFVM mesh = makeTriReferenceMesh();
//  ASSERT(mesh.vertexFaceOffsets().empty());
//  ASSERT(mesh.vertexFaceConn().empty());
//  mesh.populateVF();
//  um2::Vector<I> const vf_offsets_ref = {0, 2, 3, 5, 6};
//  um2::Vector<I> const vf_ref = {0, 1, 0, 0, 1, 1};
//  ASSERT(mesh.vertexFaceOffsets() == vf_offsets_ref);
//  ASSERT(mesh.vertexFaceConn() == vf_ref);
//}
//
//TEST_CASE(intersect)
//{
//  um2::TriFVM const mesh = makeTriReferenceMesh();
//  um2::Ray2 const ray({castIfNot<Float>(0), castIfNot<Float>(0.5)}, {1, 0});
//  um2::Vector<Float> intersections;
//  mesh.intersect(ray, intersections);
//  ASSERT(intersections.size() == 4);
//  ASSERT_NEAR(intersections[0], castIfNot<Float>(0), eps);
//  ASSERT_NEAR(intersections[1], castIfNot<Float>(0.5), eps);
//  ASSERT_NEAR(intersections[2], castIfNot<Float>(0.5), eps);
//  ASSERT_NEAR(intersections[3], castIfNot<Float>(1), eps);
//}
//
////// template <std::floating_point T, std::signed_integral I>
////// TEST_CASE(toPolytopeSoup)
//////{
//////   um2::TriFVM const tri_mesh = makeTriReferenceMesh();
//////   um2::PolytopeSoup<T, I> tri_poly_soup_ref;
//////   makeReferenceTriPolytopeSoup(tri_poly_soup_ref);
//////   um2::PolytopeSoup<T, I> tri_poly_soup;
//////   tri_mesh.toPolytopeSoup(tri_poly_soup);
//////   ASSERT(tri_poly_soup.compareTo(tri_poly_soup_ref) == 10);
//////   ASSERT(tri_poly_soup.getMeshType() == um2::MeshType::Tri);
////// }
//////

#if UM2_USE_CUDA
MAKE_CUDA_KERNEL(accessors)
#endif

TEST_SUITE(TriFVM)
{
  TEST_HOSTDEV(accessors);
  TEST(addVertex_addFace);
//  TEST(poly_soup_constructor);
  TEST(boundingBox);
  TEST(faceContaining);
//  TEST(populateVF);
//  TEST(intersect);
  //  TEST((toPolytopeSoup<T, I>));
}

auto
main() -> int
{
  RUN_SUITE(TriFVM);
  return 0;
}
