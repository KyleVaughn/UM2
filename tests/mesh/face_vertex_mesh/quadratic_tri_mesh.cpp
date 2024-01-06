#include <um2/mesh/face_vertex_mesh.hpp>

#include "../helpers/setup_mesh.hpp"
#include "../helpers/setup_polytope_soup.hpp"

#include "../../test_macros.hpp"

template <std::floating_point T, std::signed_integral I>
HOSTDEV
TEST_CASE(accessors)
{
  um2::QuadraticTriMesh<2, T, I> const mesh = makeTri6ReferenceMesh<2, T, I>();
  ASSERT(mesh.numVertices() == 9);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::QuadraticTriangle<2, T> tri0_ref(mesh.getVertex(0), mesh.getVertex(1),
                                        mesh.getVertex(2), mesh.getVertex(3),
                                        mesh.getVertex(4), mesh.getVertex(5));
  auto const tri0 = mesh.getFace(0);
  ASSERT(um2::isApprox(tri0[0], tri0_ref[0]));
  ASSERT(um2::isApprox(tri0[1], tri0_ref[1]));
  ASSERT(um2::isApprox(tri0[2], tri0_ref[2]));
  ASSERT(um2::isApprox(tri0[3], tri0_ref[3]));
  ASSERT(um2::isApprox(tri0[4], tri0_ref[4]));
  ASSERT(um2::isApprox(tri0[5], tri0_ref[5]));
  um2::QuadraticTriangle<2, T> tri1_ref(mesh.getVertex(1), mesh.getVertex(6),
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

template <std::floating_point T, std::signed_integral I>
TEST_CASE(poly_soup_constructor)
{
  um2::PolytopeSoup<T, I> poly_soup;
  makeReferenceTri6PolytopeSoup(poly_soup);
  um2::QuadraticTriMesh<2, T, I> const mesh_ref = makeTri6ReferenceMesh<2, T, I>();
  um2::QuadraticTriMesh<2, T, I> const mesh(poly_soup);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (Size i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(um2::isApprox(mesh.getVertex(i), mesh_ref.getVertex(i)));
  }
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    auto const face_ref = mesh_ref.getFace(i);
    for (Size j = 0; j < 6; ++j) {
      ASSERT(um2::isApprox(face[j], face_ref[j]));
    }
  }
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
TEST_CASE(populateVF)
{
  um2::QuadraticTriMesh<2, T, I> mesh = makeTri6ReferenceMesh<2, T, I>();
  ASSERT(mesh.getVFOffsets().empty());
  ASSERT(mesh.getVF().empty());
  mesh.populateVF();
  um2::Vector<I> const vf_offsets_ref = {0, 1, 3, 5, 6, 8, 9, 10, 11, 12};
  um2::Vector<I> const vf_ref = {0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1};
  ASSERT(mesh.getVFOffsets() == vf_offsets_ref);
  ASSERT(mesh.getVF() == vf_ref);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(intersect)
{
  um2::QuadraticTriMesh<2, T, I> const mesh = makeTri6ReferenceMesh<2, T, I>();
  um2::Point2<T> const origin(0, 0);
  um2::Vec2<T> direction(static_cast<T>(0.7), static_cast<T>(0.5));
  direction.normalize();
  um2::Ray2<T> const ray(origin, direction);
  um2::Vector<T> intersections;
  um2::intersect(ray, mesh, intersections);
  ASSERT(intersections.size() == 5);
  T const int1 = um2::sqrt(static_cast<T>(0.74));
  T const int2 = 1 / (static_cast<T>(0.7) / int1);
  ASSERT_NEAR(intersections[0], 0, static_cast<T>(1e-6));
  ASSERT_NEAR(intersections[1], 0, static_cast<T>(1e-6));
  ASSERT_NEAR(intersections[2], int1, static_cast<T>(1e-6));
  ASSERT_NEAR(intersections[3], int1, static_cast<T>(1e-6));
  ASSERT_NEAR(intersections[4], int2, static_cast<T>(1e-6));
}

//template <std::floating_point T, std::signed_integral I>
//TEST_CASE(toPolytopeSoup)
//{
//  um2::QuadraticTriMesh<2, T, I> const quad_mesh = makeTri6ReferenceMesh<2, T, I>();
//  um2::PolytopeSoup<T, I> quad_poly_soup_ref;
//  makeReferenceTri6PolytopeSoup(quad_poly_soup_ref);
//  um2::PolytopeSoup<T, I> quad_poly_soup;
//  quad_mesh.toPolytopeSoup(quad_poly_soup);
//  ASSERT(quad_poly_soup.compareTo(quad_poly_soup_ref) == 10);
//  ASSERT(quad_poly_soup.getMeshType() == um2::MeshType::QuadraticTri);
//}
//
//#if UM2_USE_CUDA
//template <std::floating_point T, std::signed_integral I>
//MAKE_CUDA_KERNEL(accessors, T, I)
//#endif

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(QuadraticTriMesh)
{
  TEST_HOSTDEV(accessors, 1, 1, T, I);
  TEST((poly_soup_constructor<T, I>));
  TEST((boundingBox<T, I>));
  TEST((faceContaining<T, I>));
  TEST((populateVF<T, I>));
  TEST((intersect<T, I>));

//  TEST((toPolytopeSoup<T, I>));
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
