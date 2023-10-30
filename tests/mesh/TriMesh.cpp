#include <um2/mesh/FaceVertexMesh.hpp>

#include "./helpers/setup_mesh.hpp"
#include "./helpers/setup_polytope_soup.hpp"

#include "../test_macros.hpp"

template <std::floating_point T, std::signed_integral I>
TEST_CASE(poly_soup_constructor)
{
  um2::PolytopeSoup<T, I> poly_soup;
  makeReferenceTriPolytopeSoup(poly_soup);
  um2::TriMesh<2, T, I> mesh_ref = makeTriReferenceMesh<2, T, I>();
  um2::TriMesh<2, T, I> mesh(poly_soup);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (Size i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(um2::isApprox(mesh.vertices[i], mesh_ref.vertices[i]));
  }
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    for (Size j = 0; j < 3; ++j) {
      ASSERT(mesh.fv[i][j] == mesh_ref.fv[i][j]);
    }
  }
  ASSERT(mesh.vf_offsets == mesh_ref.vf_offsets);
  ASSERT(mesh.vf == mesh_ref.vf);
}

//template <std::floating_point T, std::signed_integral I>
//HOSTDEV
//TEST_CASE(accessors)
//{
//  um2::TriMesh<2, T, I> mesh = makeTriReferenceMesh<2, T, I>();
//  ASSERT(mesh.numVertices() == 4);
//  ASSERT(mesh.numFaces() == 2);
//  // face
//  um2::Triangle<2, T> tri0_ref(mesh.vertices[0], mesh.vertices[1], mesh.vertices[2]);
//  auto const tri0 = mesh.getFace(0);
//  ASSERT(um2::isApprox(tri0[0], tri0_ref[0]));
//  ASSERT(um2::isApprox(tri0[1], tri0_ref[1]));
//  ASSERT(um2::isApprox(tri0[2], tri0_ref[2]));
//  um2::Triangle<2, T> tri1_ref(mesh.vertices[2], mesh.vertices[3], mesh.vertices[0]);
//  auto const tri1 = mesh.getFace(1);
//  ASSERT(um2::isApprox(tri1[0], tri1_ref[0]));
//  ASSERT(um2::isApprox(tri1[1], tri1_ref[1]));
//  ASSERT(um2::isApprox(tri1[2], tri1_ref[2]));
//}
//
//template <std::floating_point T, std::signed_integral I>
//TEST_CASE(boundingBox)
//{
//  um2::TriMesh<2, T, I> const mesh = makeTriReferenceMesh<2, T, I>();
//  auto const box = mesh.boundingBox();
//  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-6));
//  ASSERT_NEAR(box.xMax(), static_cast<T>(1), static_cast<T>(1e-6));
//  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-6));
//  ASSERT_NEAR(box.yMax(), static_cast<T>(1), static_cast<T>(1e-6));
//}
//
//template <std::floating_point T, std::signed_integral I>
//TEST_CASE(intersect)
//{
//  um2::TriMesh<2, T, I> const mesh = makeTriReferenceMesh<2, T, I>();
//  um2::Ray2<T> const ray({static_cast<T>(0), static_cast<T>(0.5)}, {1, 0});
//  um2::Vector<T> intersections(10);
//  Size n = intersections.size();
//  mesh.intersect(ray, intersections.data(), &n);
//  ASSERT(n == 4);
//  ASSERT_NEAR(intersections[0], static_cast<T>(0), static_cast<T>(1e-6));
//  ASSERT_NEAR(intersections[1], static_cast<T>(0.5), static_cast<T>(1e-6));
//  ASSERT_NEAR(intersections[2], static_cast<T>(0.5), static_cast<T>(1e-6));
//  ASSERT_NEAR(intersections[3], static_cast<T>(1), static_cast<T>(1e-6));
//}
//
//template <std::floating_point T, std::signed_integral I>
//TEST_CASE(faceContaining)
//{
//  um2::TriMesh<2, T, I> const mesh = makeTriReferenceMesh<2, T, I>();
//  um2::Point2<T> p(static_cast<T>(0.5), static_cast<T>(0.25));
//  ASSERT(mesh.faceContaining(p) == 0);
//  p = um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.75));
//  ASSERT(mesh.faceContaining(p) == 1);
//}
//
//template <std::floating_point T, std::signed_integral I>
//TEST_CASE(toPolytopeSoup)
//{
//  um2::TriMesh<2, T, I> const tri_mesh = makeTriReferenceMesh<2, T, I>();
//  um2::PolytopeSoup<T, I> tri_poly_soup_ref;
//  makeReferenceTriPolytopeSoup(tri_poly_soup_ref);
//  um2::PolytopeSoup<T, I> tri_poly_soup;
//  tri_mesh.toPolytopeSoup(tri_poly_soup);
//  ASSERT(um2::compareGeometry(tri_poly_soup, tri_poly_soup_ref) == 0);
//  ASSERT(um2::compareTopology(tri_poly_soup, tri_poly_soup_ref) == 0);
//  ASSERT(tri_poly_soup.getMeshType() == um2::MeshType::Tri);
//}

#if UM2_USE_CUDA
template <std::floating_point T, std::signed_integral I>
MAKE_CUDA_KERNEL(accessors, T, I)
#endif

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(TriMesh)
{
  TEST((poly_soup_constructor<T, I>));
//  TEST_HOSTDEV(accessors, 1, 1, T, I);
//  TEST((boundingBox<T, I>));
//  TEST((intersect<T, I>));
//  TEST((faceContaining<T, I>));
//  TEST((toPolytopeSoup<T, I>));
}

auto
main() -> int
{
//  RUN_SUITE((TriMesh<float, int16_t>));
//  RUN_SUITE((TriMesh<float, int32_t>));
//  RUN_SUITE((TriMesh<float, int64_t>));
//  RUN_SUITE((TriMesh<double, int16_t>));
//  RUN_SUITE((TriMesh<double, int32_t>));
  RUN_SUITE((TriMesh<double, int64_t>));
  return 0;
}
