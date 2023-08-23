#include <um2/mesh/FaceVertexMesh.hpp>

#include "./helpers/setup_mesh.hpp"
#include "./helpers/setup_mesh_file.hpp"

#include "../test_macros.hpp"

template <std::floating_point T, std::signed_integral I>
TEST_CASE(mesh_file_constructor)
{
  um2::MeshFile<T, I> mesh_file;
  makeReferenceQuadMeshFile(mesh_file);
  um2::QuadMesh<2, T, I> mesh_ref = makeQuadReferenceMesh<2, T, I>();
  um2::QuadMesh<2, T, I> mesh(mesh_file);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (Size i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(um2::isApprox(mesh.vertices[i], mesh_ref.vertices[i]));
  }
  for (Size i = 0; i < mesh.numFaces(); ++i) {
    for (Size j = 0; j < 4; ++j) {
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
  um2::QuadMesh<2, T, I> mesh = makeQuadReferenceMesh<2, T, I>();
  ASSERT(mesh.numVertices() == 6);
  ASSERT(mesh.numFaces() == 2);
  // face
  um2::Quadrilateral<2, T> quad0_ref(mesh.vertices[0], mesh.vertices[1], mesh.vertices[2],
                                     mesh.vertices[3]);
  auto const quad0 = mesh.getFace(0);
  ASSERT(um2::isApprox(quad0[0], quad0_ref[0]));
  ASSERT(um2::isApprox(quad0[1], quad0_ref[1]));
  ASSERT(um2::isApprox(quad0[2], quad0_ref[2]));
  ASSERT(um2::isApprox(quad0[3], quad0_ref[3]));
  um2::Quadrilateral<2, T> quad1_ref(mesh.vertices[1], mesh.vertices[4], mesh.vertices[5],
                                     mesh.vertices[2]);
  auto const quad1 = mesh.getFace(1);
  ASSERT(um2::isApprox(quad1[0], quad1_ref[0]));
  ASSERT(um2::isApprox(quad1[1], quad1_ref[1]));
  ASSERT(um2::isApprox(quad1[2], quad1_ref[2]));
  ASSERT(um2::isApprox(quad1[3], quad1_ref[3]));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(boundingBox)
{
  um2::QuadMesh<2, T, I> const mesh = makeQuadReferenceMesh<2, T, I>();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.xMax(), static_cast<T>(2), static_cast<T>(1e-6));
  ASSERT_NEAR(box.yMin(), static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(box.yMax(), static_cast<T>(1), static_cast<T>(1e-6));
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(faceContaining)
{
  um2::QuadMesh<2, T, I> const mesh = makeQuadReferenceMesh<2, T, I>();
  um2::Point2<T> p(static_cast<T>(0.5), static_cast<T>(0.25));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2<T>(static_cast<T>(1.5), static_cast<T>(0.75));
  ASSERT(mesh.faceContaining(p) == 1);
}

template <std::floating_point T, std::signed_integral I>
TEST_CASE(toMeshFile)
{
  um2::QuadMesh<2, T, I> const quad_mesh = makeQuadReferenceMesh<2, T, I>();
  um2::MeshFile<T, I> quad_mesh_file_ref;
  makeReferenceQuadMeshFile(quad_mesh_file_ref);
  um2::MeshFile<T, I> quad_mesh_file;
  quad_mesh.toMeshFile(quad_mesh_file);
  ASSERT(um2::compareGeometry(quad_mesh_file, quad_mesh_file_ref) == 0);
  ASSERT(um2::compareTopology(quad_mesh_file, quad_mesh_file_ref) == 0);
  ASSERT(quad_mesh_file.getMeshType() == um2::MeshType::Quad);
}

#if UM2_USE_CUDA
template <std::floating_point T, std::signed_integral I>
MAKE_CUDA_KERNEL(accessors, T, I)
#endif

template <std::floating_point T, std::signed_integral I>
TEST_SUITE(QuadMesh)
{
  TEST((mesh_file_constructor<T, I>));
  TEST_HOSTDEV(accessors, 1, 1, T, I);
  TEST((boundingBox<T, I>));
  TEST((faceContaining<T, I>));
  TEST((toMeshFile<T, I>));
}

auto
main() -> int
{
  RUN_SUITE((QuadMesh<float, int16_t>));
  RUN_SUITE((QuadMesh<float, int32_t>));
  RUN_SUITE((QuadMesh<float, int64_t>));
  RUN_SUITE((QuadMesh<double, int16_t>));
  RUN_SUITE((QuadMesh<double, int32_t>));
  RUN_SUITE((QuadMesh<double, int64_t>));
  return 0;
}
