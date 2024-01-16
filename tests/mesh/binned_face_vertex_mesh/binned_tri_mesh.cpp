#include <um2/mesh/binned_face_vertex_mesh.hpp>

#include "../helpers/setup_mesh.hpp"
#include "../helpers/setup_polytope_soup.hpp"

#include "../../test_macros.hpp"

template <Size D, std::floating_point T, std::signed_integral I>
auto
makeBinnedTriReferenceMesh() -> um2::BinnedTriMesh<D, T, I>
{
  // Mesh
  // 6 ------- 7 ------- 8
  // |  \   5  |  \   7  |
  // |    \    |    \    |
  // |  4   \  |  6   \  |
  // 3 ------- 4 ------- 5
  // |  \   1  |  \   3  |
  // |    \    |    \    |
  // |  0   \  |  2   \  |
  // 0 ------- 1 ------- 2
  //
  // Grid
  // + ------- + ------- +
  // |         |         |
  // |         |         |
  // |         |         |
  // +    0    +    1    +
  // |         |         |
  // |         |         |
  // |         |         |
  // + ------- + ------- +
  //
  // children: 0, 4, 8
  // face_ids: 0, 1, 4, 5, 2, 3, 6, 7

  um2::Vector<um2::Point<D, T>> const v = {
      {0, 0},
      {1, 0},
      {2, 0},
      {0, 2},
      {1, 2},
      {2, 2},
      {0, 3},
      {1, 3},
      {2, 3}
  };
  um2::Vector<um2::Vec<3, I>> const fv = {
      {0, 1, 3},
      {1, 4, 3},
      {1, 2, 4},
      {2, 5, 4},
      {3, 4, 6},
      {4, 7, 6},
      {4, 5, 7},
      {5, 8, 7}
  };
  um2::TriMesh<D, T, I> const mesh(v, fv);
  return um2::BinnedTriMesh<D, T, I>(mesh);
}

template <std::floating_point T, std::signed_integral I>
HOSTDEV
TEST_CASE(accessors)
{
  um2::BinnedTriMesh<2, T, I> const mesh = makeBinnedTriReferenceMesh<2, T, I>();
  ASSERT(mesh.numVertices() == 9);
  ASSERT(mesh.numFaces() == 8);
  // face
  um2::Triangle<2, T> tri0_ref(mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(3));
  auto const tri0 = mesh.getFace(0);
  ASSERT(um2::isApprox(tri0[0], tri0_ref[0]));
  ASSERT(um2::isApprox(tri0[1], tri0_ref[1]));
  ASSERT(um2::isApprox(tri0[2], tri0_ref[2]));
  um2::Triangle<2, T> tri1_ref(mesh.getVertex(1), mesh.getVertex(4), mesh.getVertex(3));
  auto const tri1 = mesh.getFace(1);
  ASSERT(um2::isApprox(tri1[0], tri1_ref[0]));
  ASSERT(um2::isApprox(tri1[1], tri1_ref[1]));
  ASSERT(um2::isApprox(tri1[2], tri1_ref[2]));
  // grid
  auto const num_cells = mesh.gridNumCells();
  ASSERT(num_cells[0] == 2);
  ASSERT(num_cells[1] == 1);
  ASSERT(mesh.getFlatGridIndex(0, 0) == 0);
  ASSERT(mesh.getFlatGridIndex(1, 0) == 1);
  auto const face_ids = mesh.getFaceIDsInBox(0, 0);
  I const * face_start = face_ids[0];
  I const * face_end = face_ids[1];
  ASSERT(face_end - face_start == 4);
  ASSERT(*face_start == 0);
  face_start++;
  ASSERT(*face_start == 1);
  face_start++;
  ASSERT(*face_start == 4);
  face_start++;
  ASSERT(*face_start == 5);
  face_start++;

  auto const face_ids1 = mesh.getFaceIDsInBox(1, 0);
  I const * face_start1 = face_ids1[0];
  I const * face_end1 = face_ids1[1];
  ASSERT(face_end1 - face_start1 == 4);
  ASSERT(*face_start1 == 2);
  face_start1++;
  ASSERT(*face_start1 == 3);
  face_start1++;
  ASSERT(*face_start1 == 6);
  face_start1++;
  ASSERT(*face_start1 == 7);
}

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
//TEST_CASE(populateVF)
//{
//  um2::TriMesh<2, T, I> mesh = makeTriReferenceMesh<2, T, I>();
//  ASSERT(mesh.getVFOffsets().empty());
//  ASSERT(mesh.getVF().empty());
//  mesh.populateVF();
//  um2::Vector<I> const vf_offsets_ref = {0, 2, 3, 5, 6};
//  um2::Vector<I> const vf_ref = {0, 1, 0, 0, 1, 1};
//  ASSERT(mesh.getVFOffsets() == vf_offsets_ref);
//  ASSERT(mesh.getVF() == vf_ref);
//}
//
//template <std::floating_point T, std::signed_integral I>
//TEST_CASE(intersect)
//{
//  um2::TriMesh<2, T, I> const mesh = makeTriReferenceMesh<2, T, I>();
//  um2::Ray2<T> const ray({static_cast<T>(0), static_cast<T>(0.5)}, {1, 0});
//  um2::Vector<T> intersections;
//  um2::intersect(ray, mesh, intersections);
//  ASSERT(intersections.size() == 4);
//  ASSERT_NEAR(intersections[0], static_cast<T>(0), static_cast<T>(1e-6));
//  ASSERT_NEAR(intersections[1], static_cast<T>(0.5), static_cast<T>(1e-6));
//  ASSERT_NEAR(intersections[2], static_cast<T>(0.5), static_cast<T>(1e-6));
//  ASSERT_NEAR(intersections[3], static_cast<T>(1), static_cast<T>(1e-6));
//}
//
////template <std::floating_point T, std::signed_integral I>
////TEST_CASE(toPolytopeSoup)
////{
////  um2::TriMesh<2, T, I> const tri_mesh = makeTriReferenceMesh<2, T, I>();
////  um2::PolytopeSoup<T, I> tri_poly_soup_ref;
////  makeReferenceTriPolytopeSoup(tri_poly_soup_ref);
////  um2::PolytopeSoup<T, I> tri_poly_soup;
////  tri_mesh.toPolytopeSoup(tri_poly_soup);
////  ASSERT(tri_poly_soup.compareTo(tri_poly_soup_ref) == 10);
////  ASSERT(tri_poly_soup.getMeshType() == um2::MeshType::Tri);
////}
////
//#if UM2_USE_CUDA
//template <std::floating_point T, std::signed_integral I>
//MAKE_CUDA_KERNEL(accessors, T, I)
//#endif

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(BinnedTriMesh)
{
  TEST_HOSTDEV(accessors, 1, 1, T, I);
//  TEST((addVertex_addFace<T, I>));
//  TEST((poly_soup_constructor<T, I>));
//  TEST((boundingBox<T, I>));
//  TEST((faceContaining<T, I>));
//  TEST((populateVF<T, I>));
//  TEST((intersect<T, I>));
  //  TEST((toPolytopeSoup<T, I>));
}

auto
main() -> int
{
//  RUN_SUITE((BinnedTriMesh<float, int16_t>));
//  RUN_SUITE((BinnedTriMesh<float, int32_t>));
//  RUN_SUITE((BinnedTriMesh<float, int64_t>));
//  RUN_SUITE((BinnedTriMesh<double, int16_t>));
//  RUN_SUITE((BinnedTriMesh<double, int32_t>));
  RUN_SUITE((BinnedTriMesh<double, int64_t>));
  return 0;
}
