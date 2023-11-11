#include <um2/mesh/face_vertex_mesh.hpp>

#include "./helpers/setup_mesh.hpp"
#include "./helpers/setup_polytope_soup.hpp"

#include "../test_macros.hpp"

template <std::floating_point T, std::signed_integral I>
TEST_CASE(poly_soup_constructor)
{
  um2::PolytopeSoup<T, I> poly_soup;
  makeReferenceTri6PolytopeSoup(poly_soup);
  um2::QuadraticTriMesh<2, T, I> mesh_ref = makeTri6ReferenceMesh<2, T, I>();
  um2::QuadraticTriMesh<2, T, I> mesh(poly_soup);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (Size i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(um2::isApprox(mesh.vertices[i], mesh_ref.vertices[i]));
  }
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    for (Size j = 0; j < 6; ++j) {
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
  um2::QuadraticTriMesh<2, T, I> mesh = makeTri6ReferenceMesh<2, T, I>();
  ASSERT(mesh.numVertices() == 9);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::QuadraticTriangle<2, T> tri0_ref(mesh.vertices[0], mesh.vertices[1],
                                        mesh.vertices[2], mesh.vertices[3],
                                        mesh.vertices[4], mesh.vertices[5]);
  auto const tri0 = mesh.getFace(0);
  ASSERT(um2::isApprox(tri0[0], tri0_ref[0]));
  ASSERT(um2::isApprox(tri0[1], tri0_ref[1]));
  ASSERT(um2::isApprox(tri0[2], tri0_ref[2]));
  ASSERT(um2::isApprox(tri0[3], tri0_ref[3]));
  ASSERT(um2::isApprox(tri0[4], tri0_ref[4]));
  ASSERT(um2::isApprox(tri0[5], tri0_ref[5]));
  um2::QuadraticTriangle<2, T> tri1_ref(mesh.vertices[1], mesh.vertices[6],
                                        mesh.vertices[2], mesh.vertices[7],
                                        mesh.vertices[8], mesh.vertices[4]);
  auto const tri1 = mesh.getFace(1);
  ASSERT(um2::isApprox(tri1[0], tri1_ref[0]));
  ASSERT(um2::isApprox(tri1[1], tri1_ref[1]));
  ASSERT(um2::isApprox(tri1[2], tri1_ref[2]));
  ASSERT(um2::isApprox(tri1[3], tri1_ref[3]));
  ASSERT(um2::isApprox(tri1[4], tri1_ref[4]));
  ASSERT(um2::isApprox(tri1[5], tri1_ref[5]));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(boundingBox)
{
  um2::QuadraticTriMesh<2, T, I> const mesh = makeTri6ReferenceMesh<2, T, I>();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.xMax(), static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.yMax(), static_cast<T>(1), static_cast<T>(1e-6));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(faceContaining)
{
  um2::QuadraticTriMesh<2, T, I> const mesh = makeTri6ReferenceMesh<2, T, I>();
  um2::Point2<T> p(static_cast<T>(0.6), static_cast<T>(0.5));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2<T>(static_cast<T>(0.8), static_cast<T>(0.5));
  ASSERT(mesh.faceContaining(p) == 1);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(toPolytopeSoup)
{
  um2::QuadraticTriMesh<2, T, I> const quad_mesh = makeTri6ReferenceMesh<2, T, I>();
  um2::PolytopeSoup<T, I> quad_poly_soup_ref;
  makeReferenceTri6PolytopeSoup(quad_poly_soup_ref);
  um2::PolytopeSoup<T, I> quad_poly_soup;
  quad_mesh.toPolytopeSoup(quad_poly_soup);
  ASSERT(um2::compareGeometry(quad_poly_soup, quad_poly_soup_ref) == 0);
  ASSERT(um2::compareTopology(quad_poly_soup, quad_poly_soup_ref) == 0);
  ASSERT(quad_poly_soup.getMeshType() == um2::MeshType::QuadraticTri);
}

#if UM2_USE_CUDA
template <std::floating_point T, std::signed_integral I>
MAKE_CUDA_KERNEL(accessors, T, I)
#endif

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(QuadraticTriMesh)
{
  TEST((poly_soup_constructor<T, I>));
  TEST_HOSTDEV(accessors, 1, 1, T, I);
  TEST((boundingBox<T, I>));
  TEST((faceContaining<T, I>));
  TEST((toPolytopeSoup<T, I>));
}

auto
main() -> int
{
  RUN_SUITE((QuadraticTriMesh<float, int16_t>));
  RUN_SUITE((QuadraticTriMesh<float, int32_t>));
  RUN_SUITE((QuadraticTriMesh<float, int64_t>));
  RUN_SUITE((QuadraticTriMesh<double, int16_t>));
  RUN_SUITE((QuadraticTriMesh<double, int32_t>));
  RUN_SUITE((QuadraticTriMesh<double, int64_t>));
  return 0;
}
