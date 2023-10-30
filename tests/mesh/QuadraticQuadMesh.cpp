#include <um2/mesh/FaceVertexMesh.hpp>

#include "./helpers/setup_mesh.hpp"
#include "./helpers/setup_polytope_soup.hpp"

#include "../test_macros.hpp"

template <std::floating_point T, std::signed_integral I>
TEST_CASE(poly_soup_constructor)
{
  um2::PolytopeSoup<T, I> poly_soup;
  makeReferenceQuad8PolytopeSoup(poly_soup);
  um2::QuadraticQuadMesh<2, T, I> mesh_ref = makeQuad8ReferenceMesh<2, T, I>();
  um2::QuadraticQuadMesh<2, T, I> mesh(poly_soup);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (Size i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(um2::isApprox(mesh.vertices[i], mesh_ref.vertices[i]));
  }
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    for (Size j = 0; j < 8; ++j) {
      ASSERT(mesh.fv[i][j] == mesh_ref.fv[i][j]);
    }
  }
  ASSERT(mesh.vf_offsets == mesh_ref.vf_offsets);
  ASSERT(mesh.vf == mesh_ref.vf);
}

template <std::floating_point T, std::signed_integral I>
HOSTDEV
TEST_CASE(accessors)
{
  um2::QuadraticQuadMesh<2, T, I> mesh = makeQuad8ReferenceMesh<2, T, I>();
  ASSERT(mesh.numVertices() == 13);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::QuadraticQuadrilateral<2, T> quad0_ref(
      mesh.vertices[0], mesh.vertices[1], mesh.vertices[2], mesh.vertices[3],
      mesh.vertices[6], mesh.vertices[7], mesh.vertices[8], mesh.vertices[9]);
  auto const quad0 = mesh.getFace(0);
  ASSERT(um2::isApprox(quad0[0], quad0_ref[0]));
  ASSERT(um2::isApprox(quad0[1], quad0_ref[1]));
  ASSERT(um2::isApprox(quad0[2], quad0_ref[2]));
  ASSERT(um2::isApprox(quad0[3], quad0_ref[3]));
  ASSERT(um2::isApprox(quad0[4], quad0_ref[4]));
  ASSERT(um2::isApprox(quad0[5], quad0_ref[5]));
  ASSERT(um2::isApprox(quad0[6], quad0_ref[6]));
  ASSERT(um2::isApprox(quad0[7], quad0_ref[7]));
  um2::QuadraticQuadrilateral<2, T> quad1_ref(
      mesh.vertices[1], mesh.vertices[4], mesh.vertices[5], mesh.vertices[2],
      mesh.vertices[10], mesh.vertices[11], mesh.vertices[12], mesh.vertices[7]);
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

template <std::floating_point T, std::signed_integral I>
TEST_CASE(boundingBox)
{
  um2::QuadraticQuadMesh<2, T, I> const mesh = makeQuad8ReferenceMesh<2, T, I>();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.xMax(), static_cast<T>(2), static_cast<T>(1e-6));
  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.yMax(), static_cast<T>(1), static_cast<T>(1e-6));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(faceContaining)
{
  um2::QuadraticQuadMesh<2, T, I> const mesh = makeQuad8ReferenceMesh<2, T, I>();
  um2::Point2<T> p(static_cast<T>(1.05), static_cast<T>(0.6));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2<T>(static_cast<T>(1.15), static_cast<T>(0.6));
  ASSERT(mesh.faceContaining(p) == 1);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(toPolytopeSoup)
{
  um2::QuadraticQuadMesh<2, T, I> const quad_mesh = makeQuadReferenceMesh<2, T, I>();
  um2::PolytopeSoup<T, I> quad_poly_soup_ref;
  makeReferenceQuad8PolytopeSoup(quad_poly_soup_ref);
  um2::PolytopeSoup<T, I> quad_poly_soup;
  quad_mesh.toPolytopeSoup(quad_poly_soup);
  ASSERT(um2::compareGeometry(quad_poly_soup, quad_poly_soup_ref) == 0);
  ASSERT(um2::compareTopology(quad_poly_soup, quad_poly_soup_ref) == 0);
  ASSERT(quad_poly_soup.getMeshType() == um2::MeshType::QuadraticQuad);
}

#if UM2_USE_CUDA
template <std::floating_point T, std::signed_integral I>
MAKE_CUDA_KERNEL(accessors, T, I)
#endif

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(QuadraticQuadMesh)
{
  TEST((poly_soup_constructor<T, I>));
  TEST_HOSTDEV(accessors, 1, 1, T, I);
  TEST((boundingBox<T, I>));
  TEST((faceContaining<T, I>));
}

auto
main() -> int
{
  RUN_SUITE((QuadraticQuadMesh<float, int16_t>));
  RUN_SUITE((QuadraticQuadMesh<float, int32_t>));
  RUN_SUITE((QuadraticQuadMesh<float, int64_t>));
  RUN_SUITE((QuadraticQuadMesh<double, int16_t>));
  RUN_SUITE((QuadraticQuadMesh<double, int32_t>));
  RUN_SUITE((QuadraticQuadMesh<double, int64_t>));
  return 0;
}
