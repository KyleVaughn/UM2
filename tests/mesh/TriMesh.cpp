#include <um2/mesh/TriMesh.hpp>

// #include "./helpers/setup_mesh_file.hpp"
#include "./helpers/setup_mesh.hpp"

#include "../test_macros.hpp"

template <Size D, std::floating_point T, std::signed_integral I>
HOSTDEV
TEST_CASE(constructor)
{
  //  um2::MeshFile<T, I> mesh_file;
  //  make_tri_reference_mesh_file(mesh_file);
  um2::TriMesh<D, T, I> ref_mesh;
  makeTriReferenceMesh(ref_mesh);
  assert(!ref_mesh.vertices.empty());
  //  um2::TriMesh<T, I> mesh(mesh_file);
  //  EXPECT_TRUE(um2::isApprox(mesh.vertices, ref_mesh.vertices));
  //  EXPECT_TRUE(mesh.fv_offsets.empty());
  //  EXPECT_EQ(mesh.fv, ref_mesh.fv);
  //  EXPECT_EQ(mesh.vf_offsets, ref_mesh.vf_offsets);
  //  EXPECT_EQ(mesh.vf, ref_mesh.vf);
}

// template <std::floating_point T, std::signed_integral I>
// TEST_CASE(boundingBox)
//{
//   um2::TriMesh<T, I> tri_mesh;
//   makeTriReferenceMesh(tri_mesh);
//   um2::AABox2<T> tri_ref_bbox = {
//       {0, 0},
//       {1, 1}
//   };
//   um2::AABox2<T> tri_bbox = boundingBox(tri_mesh);
//   EXPECT_TRUE(um2::isApprox(tri_bbox, tri_ref_bbox));
//
//   um2::QuadMesh<T, I> quad_mesh;
//   makeQuadReferenceMesh(quad_mesh);
//   um2::AABox2<T> quad_ref_bbox = {
//       {0, 0},
//       {2, 1}
//   };
//   um2::AABox2<T> quad_bbox = boundingBox(quad_mesh);
//   EXPECT_TRUE(um2::isApprox(quad_bbox, quad_ref_bbox));
//
//   um2::TriQuadMesh<T, I> tri_quad_mesh;
//   makeTriQuadReferenceMesh(tri_quad_mesh);
//   um2::AABox2<T> tri_quad_ref_bbox = {
//       {0, 0},
//       {2, 1}
//   };
//   um2::AABox2<T> tri_quad_bbox = boundingBox(tri_quad_mesh);
//   EXPECT_TRUE(um2::isApprox(tri_quad_bbox, tri_quad_ref_bbox));
//   /*
//   um2::QuadraticTriMesh<T, I> tri6_mesh;
//   make_tri6_reference_mesh(tri6_mesh);
//   um2::AABox2<T> tri6_ref_bbox = {
//       {0, 0},
//       {1, 1}
//   };
//   um2::AABox2<T> tri6_bbox = boundingBox(tri6_mesh);
//   ASSERT(um2::is_approx(tri6_bbox, tri6_ref_bbox), "bounding box");
//
//   um2::QuadraticQuadMesh<T, I> quad8_mesh;
//   make_quad8_reference_mesh(quad8_mesh);
//   um2::AABox2<T> quad8_ref_bbox = {
//       {0, 0},
//       {2, 1}
//   };
//   um2::AABox2<T> quad8_bbox = boundingBox(quad8_mesh);
//   ASSERT(um2::is_approx(quad8_bbox, quad8_ref_bbox), "bounding box");
//
//   um2::QuadraticTriQuadMesh<T, I> tri6_quad8_mesh;
//   make_tri6_quad8_reference_mesh(tri6_quad8_mesh);
//   um2::AABox2<T> tri6_quad8_ref_bbox = {
//       {0, 0},
//       {2, 1}
//   };
//   um2::AABox2<T> tri6_quad8_bbox = boundingBox(tri6_quad8_mesh);
//   ASSERT(um2::is_approx(tri6_quad8_bbox, tri6_quad8_ref_bbox), "bounding box");
//    */
// }
//
///*
// template <std::floating_point T, std::signed_integral I>
// TEST(to_mesh_file)
// um2::TriMesh<T, I> tri_mesh;
// make_tri_reference_mesh(tri_mesh);
// um2::MeshFile<T, I> tri_mesh_file_ref;
// make_tri_reference_mesh_file(tri_mesh_file_ref);
// um2::MeshFile<T, I> tri_mesh_file;
// tri_mesh.to_mesh_file(tri_mesh_file);
// EXPECT_EQ(compare_geometry(tri_mesh_file, tri_mesh_file_ref), );
// ASSERT(std::all_of(tri_mesh_file.element_types.begin(),
// tri_mesh_file.element_types.end(),
//                    [](auto & et) {
//                      return et == static_cast<int8_t>(um2::XDMFCellType::TRIANGLE);
//                    }),
//        "tri mesh file element types");
//
// um2::QuadMesh<T, I> quad_mesh;
// make_quad_reference_mesh(quad_mesh);
// um2::MeshFile<T, I> quad_mesh_file_ref;
// make_quad_reference_mesh_file(quad_mesh_file_ref);
// um2::MeshFile<T, I> quad_mesh_file;
// quad_mesh.to_mesh_file(quad_mesh_file);
// EXPECT_EQ(compare_geometry(quad_mesh_file, quad_mesh_file_ref), );
// ASSERT(std::all_of(quad_mesh_file.element_types.begin(),
//                    quad_mesh_file.element_types.end(),
//                    [](auto & et) {
//                      return et == static_cast<int8_t>(um2::XDMFCellType::QUAD);
//                    }),
//        "quad mesh file element types");
//
// um2::TriQuadMesh<T, I> tri_quad_mesh;
// make_tri_quad_reference_mesh(tri_quad_mesh);
// um2::MeshFile<T, I> tri_quad_mesh_file_ref;
// make_tri_quad_reference_mesh_file(tri_quad_mesh_file_ref);
// um2::MeshFile<T, I> tri_quad_mesh_file;
// tri_quad_mesh.to_mesh_file(tri_quad_mesh_file);
// EXPECT_EQ(compare_geometry(tri_quad_mesh_file, tri_quad_mesh_file_ref), );
// EXPECT_EQ(tri_quad_mesh_file.element_types.size(), 2);
// EXPECT_EQ(tri_quad_mesh_file.element_types[0],
//           static_cast<int8_t>(um2::XDMFCellType::QUAD));
// EXPECT_EQ(tri_quad_mesh_file.element_types[1],
//           static_cast<int8_t>(um2::XDMFCellType::TRIANGLE));
// um2::QuadraticTriMesh<T, I> tri6_mesh;
// make_tri6_reference_mesh(tri6_mesh);
// um2::MeshFile<T, I> tri6_mesh_file_ref;
// make_tri6_reference_mesh_file(tri6_mesh_file_ref);
// um2::MeshFile<T, I> tri6_mesh_file;
// tri6_mesh.to_mesh_file(tri6_mesh_file);
// EXPECT_EQ(compare_geometry(tri6_mesh_file, tri6_mesh_file_ref), );
// ASSERT(std::all_of(tri6_mesh_file.element_types.begin(),
//                    tri6_mesh_file.element_types.end(),
//                    [](auto & et) {
//                      return et ==
//                             static_cast<int8_t>(um2::XDMFCellType::QUADRATIC_TRIANGLE);
//                    }),
//        "tri6 mesh file element types");
//
// um2::QuadraticQuadMesh<T, I> quad8_mesh;
// make_quad8_reference_mesh(quad8_mesh);
// um2::MeshFile<T, I> quad8_mesh_file_ref;
// make_quad8_reference_mesh_file(quad8_mesh_file_ref);
// um2::MeshFile<T, I> quad8_mesh_file;
// quad8_mesh.to_mesh_file(quad8_mesh_file);
// EXPECT_EQ(compare_geometry(quad8_mesh_file, quad8_mesh_file_ref), );
// ASSERT(std::all_of(quad8_mesh_file.element_types.begin(),
//                    quad8_mesh_file.element_types.end(),
//                    [](auto & et) {
//                      return et ==
//                      static_cast<int8_t>(um2::XDMFCellType::QUADRATIC_QUAD);
//                    }),
//        "quad8 mesh file element types");
//
// um2::QuadraticTriQuadMesh<T, I> tri6_quad8_mesh;
// make_tri6_quad8_reference_mesh(tri6_quad8_mesh);
// um2::MeshFile<T, I> tri6_quad8_mesh_file_ref;
// make_tri6_quad8_reference_mesh_file(tri6_quad8_mesh_file_ref);
// um2::MeshFile<T, I> tri6_quad8_mesh_file;
// tri6_quad8_mesh.to_mesh_file(tri6_quad8_mesh_file);
// EXPECT_EQ(compare_geometry(tri6_quad8_mesh_file, tri6_quad8_mesh_file_ref), );
// EXPECT_EQ(tri6_quad8_mesh_file.element_types.size(), 2);
// ASSERT(tri6_quad8_mesh_file.element_types[0] ==
//            static_cast<int8_t>(um2::XDMFCellType::QUADRATIC_QUAD),
//        "tri6 quad8 mesh file element types");
// ASSERT(tri6_quad8_mesh_file.element_types[1] ==
//            static_cast<int8_t>(um2::XDMFCellType::QUADRATIC_TRIANGLE),
//        "tri6 quad8 mesh file element types");
// }
//  */
////
//// #if UM2_HAS_CUDA
//// template <std::floating_point T, std::signed_integral I>
//// ADD_TEMPLATED_CUDA_KERNEL(boundingBox, boundingBox_kernel, T, I)
//// template <std::floating_point T, std::signed_integral I>
//// ADD_TEMPLATED_KERNEL_TEST(boundingBox_kernel, boundingBox_cuda, T, I)
//// #endif
////
#if UM2_ENABLE_CUDA
template <Size D, std::floating_point T, std::signed_integral I>
MAKE_CUDA_KERNEL(constructor, D, T, I);
#endif

template <Size D, std::floating_point T, std::signed_integral I>
TEST_SUITE(TriMesh)
{
  TEST_HOSTDEV(constructor, 1, 1, D, T, I);
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
