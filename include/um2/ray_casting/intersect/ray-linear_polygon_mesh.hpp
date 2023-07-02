#pragma once

#include <cassert>
#include <um2/mesh/face_vertex_mesh.hpp>
#include <um2/ray_casting/intersect/ray-line_segment.hpp>

namespace um2
{

// template <std::floating_point T, std::signed_integral I>
////UM2_HOT
// void intersect(Ray2<T> const & ray,
//                TriMesh<T, I> const & mesh,
//                Vector<T> & intersections)
//{
//   T const r_miss = INF_POINT<T>;
//   intersections.resize(0); // Doesn't change capacity
//   assert(intersections.capacity() >= 0);
//   for (length_t iface = 0; iface < num_faces(mesh); ++iface) {
//       auto const & v0 = mesh.vertices[mesh.fv[3 * iface    ]];
//       auto const & v1 = mesh.vertices[mesh.fv[3 * iface + 1]];
//       auto const & v2 = mesh.vertices[mesh.fv[3 * iface + 2]];
//       T const r0 = intersect(ray, LineSegment2<T>(v0, v1));
//       T const r1 = intersect(ray, LineSegment2<T>(v1, v2));
//       T const r2 = intersect(ray, LineSegment2<T>(v2, v0));
//       if (r0 < r_miss) {
//         intersections.push_back(r0);
//       }
//       if (r1 < r_miss) {
//         intersections.push_back(r1);
//       }
//       if (r2 < r_miss) {
//         intersections.push_back(r2);
//       }
//   }
// }

// Fixed size buffer
template <std::floating_point T, std::signed_integral I>
// UM2_HOT
void intersect(
    Ray2<T> const & ray, TriMesh<T, I> const & mesh, T * const intersections,
    int * const n) // length of intersections on input, number of intersections on output
{
  T const r_miss = infiniteDistance<T>();
  int nintersect = 0;
  for (len_t iface = 0; iface < numFaces(mesh); ++iface) {
    auto const & v0 = mesh.vertices[mesh.fv[3 * iface]];
    auto const & v1 = mesh.vertices[mesh.fv[3 * iface + 1]];
    auto const & v2 = mesh.vertices[mesh.fv[3 * iface + 2]];
    T const r0 = intersect(ray, LineSegment2<T>(v0, v1));
    if (r0 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r0;
    }
    T const r1 = intersect(ray, LineSegment2<T>(v1, v2));
    if (r1 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r1;
    }
    T const r2 = intersect(ray, LineSegment2<T>(v2, v0));
    if (r2 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r2;
    }
  }
  *n = nintersect;
}

// Fixed size buffer
template <std::floating_point T, std::signed_integral I>
// UM2_HOT
void intersect(
    Ray2<T> const & ray, QuadMesh<T, I> const & mesh, T * const intersections,
    int * const n) // length of intersections on input, number of intersections on output
{
  T const r_miss = infiniteDistance<T>();
  int nintersect = 0;
  for (len_t iface = 0; iface < numFaces(mesh); ++iface) {
    auto const & v0 = mesh.vertices[mesh.fv[4 * iface]];
    auto const & v1 = mesh.vertices[mesh.fv[4 * iface + 1]];
    auto const & v2 = mesh.vertices[mesh.fv[4 * iface + 2]];
    auto const & v3 = mesh.vertices[mesh.fv[4 * iface + 3]];
    T const r0 = intersect(ray, LineSegment2<T>(v0, v1));
    if (r0 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r0;
    }
    T const r1 = intersect(ray, LineSegment2<T>(v1, v2));
    if (r1 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r1;
    }
    T const r2 = intersect(ray, LineSegment2<T>(v2, v3));
    if (r2 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r2;
    }
    T const r3 = intersect(ray, LineSegment2<T>(v3, v0));
    if (r3 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r3;
    }
  }
  *n = nintersect;
}

template <std::floating_point T, std::signed_integral I>
// UM2_HOT
void intersect(
    Ray2<T> const & ray, TriQuadMesh<T, I> const & mesh, T * const intersections,
    int * const n) // length of intersections on input, number of intersections on output
{
  T const r_miss = infiniteDistance<T>();
  int nintersect = 0;
  for (len_t iface = 0; iface < numFaces(mesh); ++iface) {
    auto const offset0 = mesh.fv_offsets[iface];
    auto const offset1 = mesh.fv_offsets[iface + 1];
    auto const nverts = offset1 - offset0;
    auto const & v0 = mesh.vertices[mesh.fv[offset0]];
    auto const & v1 = mesh.vertices[mesh.fv[offset0 + 1]];
    auto const & v2 = mesh.vertices[mesh.fv[offset0 + 2]];
    T const r0 = intersect(ray, LineSegment2<T>(v0, v1));
    if (r0 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r0;
    }
    T const r1 = intersect(ray, LineSegment2<T>(v1, v2));
    if (r1 < r_miss) {
      assert(nintersect < *n);
      intersections[nintersect++] = r1;
    }
    if (nverts == 3) {
      T const r2 = intersect(ray, LineSegment2<T>(v2, v0));
      if (r2 < r_miss) {
        assert(nintersect < *n);
        intersections[nintersect++] = r2;
      }
    } else { // nverts == 4
      auto const & v3 = mesh.vertices[mesh.fv[offset0 + 3]];
      T const r2 = intersect(ray, LineSegment2<T>(v2, v3));
      if (r2 < r_miss) {
        assert(nintersect < *n);
        intersections[nintersect++] = r2;
      }
      T const r3 = intersect(ray, LineSegment2<T>(v3, v0));
      if (r3 < r_miss) {
        assert(nintersect < *n);
        intersections[nintersect++] = r3;
      }
    }
  }
  *n = nintersect;
}

} // namespace um2
