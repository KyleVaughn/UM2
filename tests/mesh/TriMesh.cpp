#include <um2/mesh/TriMesh.hpp>

// #include "./helpers/setup_mesh_file.hpp"
#include "./helpers/setup_mesh.hpp"

#include "../test_macros.hpp"

template <std::floating_point T, std::signed_integral I>
HOSTDEV
TEST_CASE(accessors)
{
  um2::TriMesh<2, T, I> mesh = makeTriReferenceMesh<2, T, I>();
  ASSERT(mesh.numVertices() == 4);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::Triangle<2, T> tri0_ref(mesh.vertices[0], mesh.vertices[1], mesh.vertices[2]);
  auto const tri0 = mesh.face(0);
  ASSERT(um2::isApprox(tri0[0], tri0_ref[0]));
  ASSERT(um2::isApprox(tri0[1], tri0_ref[1]));
  ASSERT(um2::isApprox(tri0[2], tri0_ref[2]));
  um2::Triangle<2, T> tri1_ref(mesh.vertices[2], mesh.vertices[3], mesh.vertices[0]);
  auto const tri1 = mesh.face(1);
  ASSERT(um2::isApprox(tri1[0], tri1_ref[0]));
  ASSERT(um2::isApprox(tri1[1], tri1_ref[1]));
  ASSERT(um2::isApprox(tri1[2], tri1_ref[2]));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(boundingBox)
{
  um2::TriMesh<2, T, I> mesh = makeTriReferenceMesh<2, T, I>();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.xMax(), static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.yMax(), static_cast<T>(1), static_cast<T>(1e-6));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(faceContaining)
{
  um2::TriMesh<2, T, I> mesh = makeTriReferenceMesh<2, T, I>();
  um2::Point2<T> p(static_cast<T>(0.5), static_cast<T>(0.25));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2<T>(static_cast<T>(0.5), static_cast<T>(0.75));
  ASSERT(mesh.faceContaining(p) == 1);
}

//template <std::floating_point T, std::signed_integral I>
//TEST_CASE(regularPartition)
//{
//  um2::TriMesh<2, T, I> mesh = makeTriReferenceMesh<2, T, I>();
//  um2::Vector<I> face_ids_buffer;
//  // All in one cell
//  auto const partition = mesh.regularPartition(face_ids_buffer, static_cast<T>(0.001));
//  ASSERT_NEAR(partition.xMin(), static_cast<T>(0), static_cast<T>(1e-6));
//  ASSERT_NEAR(partition.xMax(), static_cast<T>(1), static_cast<T>(1e-6));
//  ASSERT_NEAR(partition.yMin(), static_cast<T>(0), static_cast<T>(1e-6));
//  ASSERT_NEAR(partition.yMax(), static_cast<T>(1), static_cast<T>(1e-6));
//  ASSERT(partition.numXCells() == 1);
//  ASSERT(partition.numYCells() == 1);
//  ASSERT_NEAR(partition.dx(), static_cast<T>(1), static_cast<T>(1e-6));
//  ASSERT_NEAR(partition.dy(), static_cast<T>(1), static_cast<T>(1e-6));
//  ASSERT(partition.children.size() == 2);
//  ASSERT(partition.children[0] == 0);
//  ASSERT(partition.children[1] == 2);
//  ASSERT(face_ids_buffer.size() == 2);
//  ASSERT(face_ids_buffer[0] == 0);
//  ASSERT(face_ids_buffer[1] == 1);
//}

#if UM2_ENABLE_CUDA
template <std::floating_point T, std::signed_integral I>
MAKE_CUDA_KERNEL(accessors, T, I)
#endif

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(TriMesh)
{
  TEST_HOSTDEV(accessors, 1, 1, T, I);
  TEST((boundingBox<T, I>));
  TEST((faceContaining<T, I>));
//  TEST((regularPartition<T, I>));
}

auto
main() -> int
{
  RUN_SUITE((TriMesh<float, int16_t>));
  RUN_SUITE((TriMesh<float, int32_t>));
  RUN_SUITE((TriMesh<float, int64_t>));
  RUN_SUITE((TriMesh<double, int16_t>));
  RUN_SUITE((TriMesh<double, int32_t>));
  RUN_SUITE((TriMesh<double, int64_t>));
  return 0;
}
