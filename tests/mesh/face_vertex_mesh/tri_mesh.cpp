#include <um2/config.hpp>
#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/mesh/polytope_soup.hpp>
#include <um2/common/logger.hpp>
#include <um2/common/cast_if_not.hpp>
#include <um2/stdlib/vector.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/point.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/math/vec.hpp>
#include <um2/geometry/polytope.hpp>

#include "../helpers/setup_mesh.hpp"
#include "../helpers/setup_polytope_soup.hpp"

#include "../../test_macros.hpp"

#include <algorithm>
#include <cstdint>
#include <random>

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
  um2::Vector<um2::Point2F> & verts = mesh.vertices();
  for (um2::Point2F & v : verts) {
    v[0] += dis(gen);
    v[1] += dis(gen);
  }
}

auto constexpr eps = um2::epsDistance<Float>();

HOSTDEV
TEST_CASE(accessors)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();

  // numVertices, numFaces
  ASSERT(mesh.numVertices() == 4);
  ASSERT(mesh.numFaces() == 2);

  // getVertex
  ASSERT(mesh.getVertex(0).isApprox(um2::Point2F(0, 0)));
  ASSERT(mesh.getVertex(1).isApprox(um2::Point2F(1, 0)));

  // getFace
  um2::Triangle2F tri0_ref(mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(2));
  auto const tri0 = mesh.getFace(0);
  ASSERT(tri0[0].isApprox(tri0_ref[0]));
  ASSERT(tri0[1].isApprox(tri0_ref[1]));
  ASSERT(tri0[2].isApprox(tri0_ref[2]));
  um2::Triangle2F tri1_ref(mesh.getVertex(2), mesh.getVertex(3), mesh.getVertex(0));
  auto const tri1 = mesh.getFace(1);
  ASSERT(tri1[0].isApprox(tri1_ref[0]));
  ASSERT(tri1[1].isApprox(tri1_ref[1]));
  ASSERT(tri1[2].isApprox(tri1_ref[2]));

  // getEdge
  auto const edge0 = mesh.getEdge(0, 0);
  ASSERT(edge0[0].isApprox(tri0[0]));
  ASSERT(edge0[1].isApprox(tri0[1]));
  auto const edge1 = mesh.getEdge(0, 1);
  ASSERT(edge1[0].isApprox(tri0[1]));
  ASSERT(edge1[1].isApprox(tri0[2]));
  auto const edge2 = mesh.getEdge(0, 2);
  ASSERT(edge2[0].isApprox(tri0[2]));
  ASSERT(edge2[1].isApprox(tri0[0]));

  // getEdgeConn
  auto const edge_conn0 = mesh.getEdgeConn(0, 0);
  ASSERT(edge_conn0[0] == 0);
  ASSERT(edge_conn0[1] == 1);
  auto const edge_conn1 = mesh.getEdgeConn(0, 1);
  ASSERT(edge_conn1[0] == 1);
  ASSERT(edge_conn1[1] == 2);
  auto const edge_conn2 = mesh.getEdgeConn(0, 2);
  ASSERT(edge_conn2[0] == 2);
  ASSERT(edge_conn2[1] == 0);
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

  ASSERT(mesh.getVertex(0).isApprox(um2::Point2F(0, 0)));
  ASSERT(mesh.getVertex(1).isApprox(um2::Point2F(1, 0)));

  // Same as reference mesh. Should make an == operator for meshes.
  um2::Triangle2F tri0_ref(mesh.getVertex(0), mesh.getVertex(1), mesh.getVertex(2));
  auto const tri0 = mesh.getFace(0);
  ASSERT(tri0[0].isApprox(tri0_ref[0]));
  ASSERT(tri0[1].isApprox(tri0_ref[1]));
  ASSERT(tri0[2].isApprox(tri0_ref[2]));
  um2::Triangle2F tri1_ref(mesh.getVertex(2), mesh.getVertex(3), mesh.getVertex(0));
  auto const tri1 = mesh.getFace(1);
  ASSERT(tri1[0].isApprox(tri1_ref[0]));
  ASSERT(tri1[1].isApprox(tri1_ref[1]));
  ASSERT(tri1[2].isApprox(tri1_ref[2]));
}

TEST_CASE(boundingBox)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();
  auto const box = mesh.boundingBox();
  ASSERT(box.minima().isApprox(um2::Point2F(0, 0)));
  ASSERT(box.maxima().isApprox(um2::Point2F(1, 1)));

  um2::TriFVM mesh2;
  makeTriangleMesh(mesh2, 2);
  auto const box2 = mesh2.boundingBox();
  ASSERT(box2.minima().isApprox(um2::Point2F(0, 0)));
  ASSERT(box2.maxima().isApprox(um2::Point2F(2, 2)));

  um2::TriFVM mesh3;
  makeTriangleMesh(mesh3, 3);
  auto const box3 = mesh3.boundingBox();
  ASSERT(box3.minima().isApprox(um2::Point2F(0, 0)));
  ASSERT(box3.maxima().isApprox(um2::Point2F(3, 3)));
}

TEST_CASE(faceContaining)
{
  um2::TriFVM const mesh = makeTriReferenceMesh();
  um2::Point2F p(castIfNot<Float>(0.5), castIfNot<Float>(0.25));
  ASSERT(mesh.faceContaining(p) == 0);
  p = um2::Point2F(castIfNot<Float>(0.5), castIfNot<Float>(0.75));
  ASSERT(mesh.faceContaining(p) == 1);

  Int const ntri = 5; // 2 * ntri * ntri triangles
  // The min coord is 0 and max coord is ntri
  Int const npoints = 1000;

  um2::Point2F const pmin(castIfNot<Float>(0.11), castIfNot<Float>(0.11));
  um2::Point2F const pmax(castIfNot<Float>(ntri - 0.11), castIfNot<Float>(ntri - 0.11));
  um2::AxisAlignedBox2F const box(pmin, pmax);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Float> dis(pmin[0], pmax[0]);
  for (Int ip = 0; ip < 10; ++ip) {
    um2::TriFVM mesh2;
    makeTriangleMesh(mesh2, ntri);
    perturb(mesh2); // maximum perturbation is 0.1
    for (Int i = 0; i < npoints; ++i) {
      um2::Point2F const pt(dis(gen), dis(gen));
      Int const iface = mesh2.faceContaining(pt);
      if (iface == -1) {
        ASSERT(!box.contains(pt));
      } else {
        auto const tri = mesh2.getFace(iface);
        auto const a = tri.area();
        auto const a0 = um2::Triangle2F(tri[0], tri[1], pt).area();
        auto const a1 = um2::Triangle2F(tri[1], tri[2], pt).area();
        auto const a2 = um2::Triangle2F(tri[2], tri[0], pt).area();
        ASSERT_NEAR(a, a0 + a1 + a2, eps);
      }
    }
  }
}

TEST_CASE(validate)
{
  // Check that clockwise faces are fixed
  // 2 ---- 3
  // | \    |
  // |    \ |
  // 0 ---- 1
  // F0 = {0, 1, 2}
  // F1 = {2, 3, 1} <- not CCW
  um2::TriFVM mesh_ccw;
  mesh_ccw.addVertex({0, 0});
  mesh_ccw.addVertex({1, 0});
  mesh_ccw.addVertex({0, 1});
  mesh_ccw.addVertex({1, 1});
  mesh_ccw.addFace({0, 1, 2});
  mesh_ccw.addFace({2, 3, 1});
  ASSERT(mesh_ccw.getFace(0).isCCW());
  ASSERT(!mesh_ccw.getFace(1).isCCW());
  mesh_ccw.validate();
  ASSERT(mesh_ccw.getFace(0).isCCW());
  ASSERT(mesh_ccw.getFace(1).isCCW());
  auto sv = um2::logger::getLastMessage();
  // Check warning message
  ASSERT(sv.find_first_of("Some faces were flipped to ensure counter-clockwise order")
      != um2::StringView::npos);

  // Check that the mesh's boundary edges form a single closed loop
  // This should account for:
  //  1. Holes
  //  2. Overlap

  // Mesh with a hole on the boundary
  // 8------9-----10-----11
  // |\   6 |\   8 |\  10 |
  // |  \   |  \   |  \   |
  // | 5  \ | 7  \ | 9  \ |
  // 4------5------6------7
  // |\   1 |\ HOLE|\   4 |
  // |  \   |  \   |  \   |
  // | 0  \ | 2  \ | 3  \ |
  // 0------1------2------3
  //
  // Boundary: (0, 1), (1, 2), (2, 3), (3, 7), (7, 11), (11, 10), (10, 9),
  //           (9, 8), (8, 4), (4, 0)
  // Interior boundary: (6, 2), (5, 6), (2, 5)
  um2::logger::exit_on_error = false;
  um2::TriFVM mesh;
  for (Int j = 0; j < 3; ++j) {
    for (Int i = 0; i <= 3; ++i) {
      mesh.addVertex({i, j});
    }
  }
  mesh.addFace({0, 1, 4});
  mesh.addFace({1, 5, 4});
  mesh.addFace({1, 2, 5});
  mesh.addFace({2, 3, 6});
  mesh.addFace({3, 7, 6});
  mesh.addFace({4, 5, 8});
  mesh.addFace({5, 9, 8});
  mesh.addFace({5, 6, 9});
  mesh.addFace({6, 10, 9});
  mesh.addFace({6, 7, 10});
  mesh.addFace({7, 11, 10});
  mesh.validate();
  sv = um2::logger::getLastMessage();
  ASSERT(sv.find_first_of("Mesh has a hole on its boundary") != um2::StringView::npos);

  // Mesh with an hole in the interior
  // 12-----13----14-----15
  // |\   12|\   14|\  16 |
  // |  \   |  \   |  \   |
  // | 11 \ |13  \ |15  \ |
  // 8------9-----10-----11
  // |\   7 |\ HOLE|\  10 |
  // |  \   |  \   |  \   |
  // | 6  \ | 8  \ | 9  \ |
  // 4------5------6------7
  // |\   1 |\   3 |\   5 |
  // |  \   |  \   |  \   |
  // | 0  \ | 2  \ | 4  \ |
  // 0------1------2------3
  //
  um2::TriFVM mesh2;
  for (Int j = 0; j <= 3; ++j) {
    for (Int i = 0; i <= 3; ++i) {
      mesh2.addVertex({i, j});
    }
  }
  mesh2.addFace({0, 1, 4}); // 0
  mesh2.addFace({1, 5, 4}); // 1
  mesh2.addFace({1, 2, 5}); // 2
  mesh2.addFace({2, 6, 5}); // 3
  mesh2.addFace({2, 3, 6}); // 4
  mesh2.addFace({3, 7, 6}); // 5
  mesh2.addFace({4, 5, 8}); // 6
  mesh2.addFace({5, 9, 8}); // 7
  mesh2.addFace({5, 6, 9}); // 8
  mesh2.addFace({6, 7, 10}); // 9
  mesh2.addFace({7, 11, 10}); // 10
  mesh2.addFace({8, 9, 12}); // 11
  mesh2.addFace({9, 13, 12}); // 12
  mesh2.addFace({9, 10, 13}); // 13
  mesh2.addFace({10, 14, 13}); // 14
  mesh2.addFace({10, 11, 14}); // 15
  mesh2.addFace({11, 15, 14}); // 16
  mesh2.validate();
  sv = um2::logger::getLastMessage();
  ASSERT(sv.find_first_of("Mesh has a hole in its interior") != um2::StringView::npos);
  um2::logger::exit_on_error = true;
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

TEST_CASE(mortonSortVertices)
{
  um2::TriFVM mesh;
  makeTriangleMesh(mesh, 2);
  um2::Vec<3, Int> face_conn = mesh.getFaceConn(0);
  ASSERT(face_conn[0] == 0);
  ASSERT(face_conn[1] == 1);
  ASSERT(face_conn[2] == 3);
  face_conn = mesh.getFaceConn(1);
  ASSERT(face_conn[0] == 4);
  ASSERT(face_conn[1] == 3);
  ASSERT(face_conn[2] == 1);
  mesh.mortonSortVertices();
  ASSERT(mesh.getVertex(0).isApprox(um2::Point2F(0, 0)));
  ASSERT(mesh.getVertex(1).isApprox(um2::Point2F(1, 0)));
  ASSERT(mesh.getVertex(2).isApprox(um2::Point2F(0, 1)));
  ASSERT(mesh.getVertex(3).isApprox(um2::Point2F(1, 1)));
  ASSERT(mesh.getVertex(4).isApprox(um2::Point2F(2, 0)));
  ASSERT(mesh.getVertex(5).isApprox(um2::Point2F(2, 1)));
  ASSERT(mesh.getVertex(6).isApprox(um2::Point2F(0, 2)));
  ASSERT(mesh.getVertex(7).isApprox(um2::Point2F(1, 2)));
  ASSERT(mesh.getVertex(8).isApprox(um2::Point2F(2, 2)));
  face_conn = mesh.getFaceConn(0);
  ASSERT(face_conn[0] == 0);
  ASSERT(face_conn[1] == 1);
  ASSERT(face_conn[2] == 2);
  face_conn = mesh.getFaceConn(1);
  ASSERT(face_conn[0] == 3);
  ASSERT(face_conn[1] == 2);
  ASSERT(face_conn[2] == 1);
}

TEST_CASE(mortonSortFaces)
{
  um2::TriFVM mesh;
  for (Int i = 0; i <= 2; ++i) {
    for (Int j = 0; j <= 2; ++j) {
      mesh.addVertex({j, i});
    }
  }
  // Post-sorting
  // 6------7------8
  // |\   5 |\   7 |
  // |  \   |  \   |
  // | 4  \ | 6  \ |
  // 3------4------5
  // |\   1 |\   3 |
  // |  \   |  \   |
  // | 0  \ | 2  \ |
  // 0------1------2

  mesh.addFace({5, 8, 7});
  mesh.addFace({4, 5, 7});
  mesh.addFace({4, 7, 6});
  mesh.addFace({3, 4, 6});
  mesh.addFace({0, 1, 3});
  mesh.addFace({1, 4, 3});
  mesh.addFace({1, 2, 4});
  mesh.addFace({2, 5, 4});

  mesh.mortonSortFaces();

  um2::Vec<3, Int> face_conn = mesh.getFaceConn(0);
  ASSERT(face_conn[0] == 0);
  ASSERT(face_conn[1] == 1);
  ASSERT(face_conn[2] == 3);
  face_conn = mesh.getFaceConn(1);
  ASSERT(face_conn[0] == 1);
  ASSERT(face_conn[1] == 4);
  ASSERT(face_conn[2] == 3);
  face_conn = mesh.getFaceConn(2);
  ASSERT(face_conn[0] == 1);
  ASSERT(face_conn[1] == 2);
  ASSERT(face_conn[2] == 4);
  face_conn = mesh.getFaceConn(3);
  ASSERT(face_conn[0] == 2);
  ASSERT(face_conn[1] == 5);
  ASSERT(face_conn[2] == 4);
  face_conn = mesh.getFaceConn(4);
  ASSERT(face_conn[0] == 3);
  ASSERT(face_conn[1] == 4);
  ASSERT(face_conn[2] == 6);
  face_conn = mesh.getFaceConn(5);
  ASSERT(face_conn[0] == 4);
  ASSERT(face_conn[1] == 7);
  ASSERT(face_conn[2] == 6);
  face_conn = mesh.getFaceConn(6);
  ASSERT(face_conn[0] == 4);
  ASSERT(face_conn[1] == 5);
  ASSERT(face_conn[2] == 7);
  face_conn = mesh.getFaceConn(7);
  ASSERT(face_conn[0] == 5);
  ASSERT(face_conn[1] == 8);
  ASSERT(face_conn[2] == 7);

  mesh.validate();
}

TEST_CASE(intersect)
{
  um2::TriFVM mesh;
  makeTriangleMesh(mesh, 2);

  // 6------7------8
  // |\   5 |\   7 |
  // |  \   |  \   |
  // | 4  \ | 6  \ |
  // 3------4------5
  // |\   1 |\   3 |
  // |  \   |  \   |
  // | 0  \ | 2  \ |
  // 0------1------2

  // Check a few basic intersections before automated testing
  um2::Point2F origin(castIfNot<Float>(-1), castIfNot<Float>(0.5));
  um2::Vec2F dir(1, 0);
  um2::Ray2F const ray(origin, dir);
  Float coords[24];
  Int offsets[24];
  Int faces[24];
  Int const hits = mesh.intersect(ray, coords);
  ASSERT(hits == 8);
  std::sort(coords, coords + hits);
  ASSERT_NEAR(coords[0], castIfNot<Float>(1.0), eps);
  ASSERT_NEAR(coords[1], castIfNot<Float>(1.5), eps);
  ASSERT_NEAR(coords[2], castIfNot<Float>(1.5), eps);
  ASSERT_NEAR(coords[3], castIfNot<Float>(2.0), eps);
  ASSERT_NEAR(coords[4], castIfNot<Float>(2.0), eps);
  ASSERT_NEAR(coords[5], castIfNot<Float>(2.5), eps);
  ASSERT_NEAR(coords[6], castIfNot<Float>(2.5), eps);
  ASSERT_NEAR(coords[7], castIfNot<Float>(3.0), eps);
  for (Int i = 0; i < hits; ++i) {
    coords[i] = 0;
  }
  auto const hits_faces = mesh.intersect(ray, coords, offsets, faces);
  ASSERT(hits_faces[0] == 8);
  ASSERT(hits_faces[1] == 4);
  for (Int i = 0; i < 4; ++i) {
    ASSERT(offsets[i] == 2 * i);
    ASSERT(faces[i] == i);
  }
  ASSERT(offsets[4] == 8);

  origin[0] = castIfNot<Float>(3);
  origin[1] = castIfNot<Float>(0.5);
  dir[0] = -1;
  dir[1] = 0;
  um2::Ray2F const ray2(origin, dir);
  Int const hits2 = mesh.intersect(ray2, coords);
  ASSERT(hits2 == 8);
  std::sort(coords, coords + hits2);
  ASSERT_NEAR(coords[0], castIfNot<Float>(1.0), eps);
  ASSERT_NEAR(coords[1], castIfNot<Float>(1.5), eps);
  ASSERT_NEAR(coords[2], castIfNot<Float>(1.5), eps);
  ASSERT_NEAR(coords[3], castIfNot<Float>(2.0), eps);
  ASSERT_NEAR(coords[4], castIfNot<Float>(2.0), eps);
  ASSERT_NEAR(coords[5], castIfNot<Float>(2.5), eps);
  ASSERT_NEAR(coords[6], castIfNot<Float>(2.5), eps);
  ASSERT_NEAR(coords[7], castIfNot<Float>(3.0), eps);
  for (Int i = 0; i < hits2; ++i) {
    coords[i] = 0;
  }

  auto const hits_faces2 = mesh.intersect(ray2, coords, offsets, faces);
  ASSERT(hits_faces2[0] == 8);
  ASSERT(hits_faces2[1] == 4);
  for (Int i = 0; i <= 4; ++i) {
    ASSERT(offsets[i] == 2 * i);
  }
  ASSERT(faces[0] == 0);
  ASSERT(faces[1] == 1);
  ASSERT(faces[2] == 2);
  ASSERT(faces[3] == 3);

  Float sorted_coords[24];
  Int sorted_offsets[24];
  Int sorted_faces[24];
  Int perm[24];
  um2::sortRayMeshIntersections(coords, offsets, faces,
                                sorted_coords, sorted_offsets, sorted_faces,
                                perm, hits_faces2);
  ASSERT(sorted_faces[0] == 3);
  ASSERT(sorted_faces[1] == 2);
  ASSERT(sorted_faces[2] == 1);
  ASSERT(sorted_faces[3] == 0);
  for (Int i = 0; i <= 4; ++i) {
    ASSERT(sorted_offsets[i] == 2 * i);
  }
  ASSERT_NEAR(sorted_coords[0], castIfNot<Float>(1.0), eps);
  ASSERT_NEAR(sorted_coords[1], castIfNot<Float>(1.5), eps);
  ASSERT_NEAR(sorted_coords[2], castIfNot<Float>(1.5), eps);
  ASSERT_NEAR(sorted_coords[3], castIfNot<Float>(2.0), eps);
  ASSERT_NEAR(sorted_coords[4], castIfNot<Float>(2.0), eps);
  ASSERT_NEAR(sorted_coords[5], castIfNot<Float>(2.5), eps);
  ASSERT_NEAR(sorted_coords[6], castIfNot<Float>(2.5), eps);
  ASSERT_NEAR(sorted_coords[7], castIfNot<Float>(3.0), eps);

}

TEST_CASE(operator_PolytopeSoup)
{
   um2::TriFVM const tri_mesh = makeTriReferenceMesh();
   um2::PolytopeSoup tri_poly_soup_ref;
   makeReferenceTriPolytopeSoup(tri_poly_soup_ref);
   um2::PolytopeSoup const tri_poly_soup = tri_mesh;
   ASSERT(tri_poly_soup.compare(tri_poly_soup_ref) == 6);
}

TEST_CASE(PolytopeSoup_constructor)
{
  um2::PolytopeSoup soup;
  makeReferenceTriPolytopeSoup(soup);
  um2::TriFVM const mesh_ref = makeTriReferenceMesh();
  um2::TriFVM const mesh(soup);
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

TEST_SUITE(TriFVM)
{
  TEST_HOSTDEV(accessors);
  TEST(addVertex_addFace);
  TEST(boundingBox);
  TEST(faceContaining);
  TEST(validate);
  TEST(populateVF);
  TEST(mortonSortVertices);
  TEST(mortonSortFaces);
  TEST(intersect);
  TEST(operator_PolytopeSoup);
  TEST(PolytopeSoup_constructor);
}

auto
main() -> int
{
  RUN_SUITE(TriFVM);
  return 0;
}
