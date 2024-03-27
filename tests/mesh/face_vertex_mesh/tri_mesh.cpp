#include <um2/mesh/face_vertex_mesh.hpp>

#include "../helpers/setup_mesh.hpp"
#include "../helpers/setup_polytope_soup.hpp"

#include "../../test_macros.hpp"

#include <random>
#include <iostream>

// Create a square mesh 2 * N^2 triangles
// +------+------+
// |\   5 |\   7 |
// |  \   |  \   |
// | 4  \ | 6  \ |
// +------+------+
// |\   1 |\   3 |
// |  \   |  \   |
// | 0  \ | 2  \ |
// +------+------+

HOSTDEV
void
makeTriangleMesh(um2::TriFVM & mesh, Int n)
{
  // Create N + 1 vertices in each direction
  for (Int i = 0; i < n + 1; ++i) {
    for (Int j = 0; j < n + 1; ++j) {
      mesh.addVertex({j, i});
    }
  }
  // Create 2 * N triangles in each row
  for (Int i = 0; i < n; ++i) {
    for (Int j = 0; j < n; ++j) {
      // v2-----v3
      // | \     |
      // |   \   |
      // |     \ |
      // v0-----v1
      Int const v0 = i * (n + 1) + j;
      Int const v1 = v0 + 1;
      Int const v2 = v0 + n + 1;
      Int const v3 = v2 + 1;
      mesh.addFace({v0, v1, v2});
      mesh.addFace({v3, v2, v1});
    }
  }
  ASSERT(mesh.numVertices() == (n + 1) * (n + 1));
  ASSERT(mesh.numFaces() == 2 * n * n);
}

void
perturb(um2::TriFVM & mesh)
{
  auto constexpr delta = castIfNot<Float>(0.1);
  uint32_t constexpr seed = 0x08FA9A20;
  // We want a fixed seed for reproducibility
  // NOLINTNEXTLINE(cert-msc32-c,cert-msc51-cpp)
  static std::mt19937 gen(seed);
  static std::uniform_real_distribution<Float> dis(-delta, delta);
  um2::Vector<um2::Point2> & verts = mesh.vertices();
  for (um2::Point2 & v : verts) {
    v[0] += dis(gen);
    v[1] += dis(gen);
  }
}

auto constexpr eps = castIfNot<Float>(1e-6);

HOSTDEV
TEST_CASE(accessors)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();

  // numVertices, numFaces
  ASSERT(mesh.numVertices() == 4);
  ASSERT(mesh.numFaces() == 2);

  // getVertex
  ASSERT(mesh.getVertex(0).isApprox(um2::Point2(0, 0)));
  ASSERT(mesh.getVertex(1).isApprox(um2::Point2(1, 0)));

  // getFace
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

  ASSERT(mesh.numVertices() == 4);
  ASSERT(mesh.numFaces() == 2);

  ASSERT(mesh.getVertex(0).isApprox(um2::Point2(0, 0)));
  ASSERT(mesh.getVertex(1).isApprox(um2::Point2(1, 0)));

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

TEST_CASE(boundingBox)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();
  auto const box = mesh.boundingBox();
  ASSERT_NEAR(box.xMin(), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.xMax(), castIfNot<Float>(1), eps);
  ASSERT_NEAR(box.yMin(), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box.yMax(), castIfNot<Float>(1), eps);

  um2::TriFVM mesh2;
  makeTriangleMesh(mesh2, 2);
  auto const box2 = mesh2.boundingBox();
  ASSERT_NEAR(box2.xMin(), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box2.xMax(), castIfNot<Float>(2), eps);
  ASSERT_NEAR(box2.yMin(), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box2.yMax(), castIfNot<Float>(2), eps);

  um2::TriFVM mesh3;
  makeTriangleMesh(mesh3, 3);
  auto const box3 = mesh3.boundingBox();
  ASSERT_NEAR(box3.xMin(), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box3.xMax(), castIfNot<Float>(3), eps);
  ASSERT_NEAR(box3.yMin(), castIfNot<Float>(0), eps);
  ASSERT_NEAR(box3.yMax(), castIfNot<Float>(3), eps);
}

TEST_CASE(PolytopeSoup_constructor)
{
  um2::PolytopeSoup poly_soup;
  makeReferenceTriPolytopeSoup(poly_soup);
  um2::TriFVM const mesh_ref = makeTriReferenceMesh();
  um2::TriFVM const mesh(poly_soup);
  ASSERT(mesh.numVertices() == mesh_ref.numVertices());
  for (Int i = 0; i < mesh.numVertices(); ++i) {
    ASSERT(mesh.getVertex(i).isApprox(mesh_ref.getVertex(i)));
  }
  for (Int i = 0; i < mesh.numFaces(); ++i) {
    auto const face = mesh.getFace(i);
    auto const face_ref = mesh_ref.getFace(i);
    for (Int j = 0; j < 3; ++j) {
      ASSERT(face[j].isApprox(face_ref[j]));
    }
  }
}

TEST_CASE(faceContaining)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();
  um2::Point2 p(castIfNot<Float>(0.5), castIfNot<Float>(0.25));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2(castIfNot<Float>(0.5), castIfNot<Float>(0.75));
  ASSERT(mesh.faceContaining(p) == 1);

  Int const ntri = 5; // 2 * ntri * ntri triangles
  // The min coord is 0 and max coord is ntri
  Int const npoints = 1000;

  um2::Point2 const pmin(castIfNot<Float>(0.11), castIfNot<Float>(0.11)); 
  um2::Point2 const pmax(castIfNot<Float>(ntri - 0.11), castIfNot<Float>(ntri - 0.11));
  um2::AxisAlignedBox2 const box(pmin, pmax);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Float> dis(pmin[0], pmax[0]);
  for (Int ip = 0; ip < 10; ++ip) {
    um2::TriFVM mesh2;
    makeTriangleMesh(mesh2, ntri);
    perturb(mesh2); // maximum perturbation is 0.1
    for (Int i = 0; i < npoints; ++i) {
      um2::Point2 const pt(dis(gen), dis(gen));
      Int const iface = mesh2.faceContaining(pt);
      if (iface == -1) {
        ASSERT(!box.contains(pt));
      } else {
        auto const tri = mesh2.getFace(iface);
        auto const a = tri.area();
        auto const a0 = um2::Triangle<2>(tri[0], tri[1], pt).area();
        auto const a1 = um2::Triangle<2>(tri[1], tri[2], pt).area();
        auto const a2 = um2::Triangle<2>(tri[2], tri[0], pt).area();
        ASSERT_NEAR(a, a0 + a1 + a2, eps);
      }
    }
  }
}

TEST_CASE(populateVF)
{
  um2::TriFVM mesh = makeTriReferenceMesh();
  ASSERT(mesh.vertexFaceOffsets().empty());
  ASSERT(mesh.vertexFaceConn().empty());
  mesh.populateVF();
  um2::Vector<Int> const vf_offsets_ref = {0, 2, 3, 5, 6};
  um2::Vector<Int> const vf_ref = {0, 1, 0, 0, 1, 1};
  ASSERT(mesh.vertexFaceOffsets() == vf_offsets_ref);
  ASSERT(mesh.vertexFaceConn() == vf_ref);
}

////TEST_CASE(intersect)
////{
////  um2::TriFVM const mesh = makeTriReferenceMesh();
////  um2::Ray2 const ray({castIfNot<Float>(0), castIfNot<Float>(0.5)}, {1, 0});
////  um2::Vector<Float> intersections;
////  mesh.intersect(ray, intersections);
////  ASSERT(intersections.size() == 4);
////  ASSERT_NEAR(intersections[0], castIfNot<Float>(0), eps);
////  ASSERT_NEAR(intersections[1], castIfNot<Float>(0.5), eps);
////  ASSERT_NEAR(intersections[2], castIfNot<Float>(0.5), eps);
////  ASSERT_NEAR(intersections[3], castIfNot<Float>(1), eps);
////}
////
//////// template <std::floating_point T, std::signed_integral I>
//////// TEST_CASE(toPolytopeSoup)
////////{
////////   um2::TriFVM const tri_mesh = makeTriReferenceMesh();
////////   um2::PolytopeSoup<T, I> tri_poly_soup_ref;
////////   makeReferenceTriPolytopeSoup(tri_poly_soup_ref);
////////   um2::PolytopeSoup<T, I> tri_poly_soup;
////////   tri_mesh.toPolytopeSoup(tri_poly_soup);
////////   ASSERT(tri_poly_soup.compareTo(tri_poly_soup_ref) == 10);
////////   ASSERT(tri_poly_soup.getMeshType() == um2::MeshType::Tri);
//////// }
////////
//
//#if UM2_USE_CUDA
//MAKE_CUDA_KERNEL(accessors)
//#endif

TEST_SUITE(TriFVM)
{
  TEST_HOSTDEV(accessors);
  TEST(addVertex_addFace);
  TEST(boundingBox);
  TEST(PolytopeSoup_constructor);
  TEST(faceContaining);
  TEST(populateVF);
//  TEST(intersect);
  //  TEST((toPolytopeSoup<T, I>));
}

auto
main() -> int
{
  RUN_SUITE(TriFVM);
  return 0;
}
