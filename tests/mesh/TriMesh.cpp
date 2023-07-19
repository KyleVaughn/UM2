#include <um2/mesh/TriMesh.hpp>

// #include "./helpers/setup_mesh_file.hpp"
#include "./helpers/setup_mesh.hpp"

#include "../test_macros.hpp"

// numVertices
// numFaces
// boundingBox
// face
// faceContaining

template <Size D, std::floating_point T, std::signed_integral I>
TEST_SUITE(TriMesh)
{
  //  TEST_HOSTDEV((boundingBox<T, I>));
  //  RUN_TEST("to_mesh_file", (to_mesh_file<T, I>));
  //
  // #if UM2_HAS_CUDA
  //    RUN_CUDA_TEST("boundingBox_cuda",   (boundingBox_cuda<T, I>) );
  // #endif
}

auto
main() -> int
{
  RUN_SUITE((TriMesh<2, float, int16_t>));
  RUN_SUITE((TriMesh<2, float, int32_t>));
  RUN_SUITE((TriMesh<2, float, int64_t>));
  RUN_SUITE((TriMesh<2, double, int16_t>));
  RUN_SUITE((TriMesh<2, double, int32_t>));
  RUN_SUITE((TriMesh<2, double, int64_t>));
  return 0;
}
