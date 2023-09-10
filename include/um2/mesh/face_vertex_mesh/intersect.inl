namespace um2
{

// template <Size N, std::floating_point T, std::signed_integral I>
// void
// intersect(PlanarLinearPolygonMesh<N, T, I> const & mesh, Ray2<T> const & ray,
//           T * intersections, Size * const n) noexcept
//{
//   T const r_miss = infiniteDistance<T>();
//   // cppcheck-suppress constStatement; justification: false positive
//   T * const start = intersections;
// #ifndef NDEBUG
//   Size const n0 = *n;
// #endif
//   for (Size i = 0; i < numFaces(mesh); ++i) {
//     auto const face = mesh.getFace(i);
//     for (Size j = 0; j < polygonNumEdges<1, N>(); ++j) {
//       auto const edge = face.getEdge(j);
//       T const r = intersect(edge, ray);
//       if (r < r_miss) {
//         assert(static_cast<Size>(intersections - start) < n0);
//         *intersections++ = r;
//       }
//     }
//   }
//   *n = static_cast<Size>(intersections - start);
//   std::sort(start, intersections);
// }

template <Size N, std::floating_point T, std::signed_integral I>
void
intersect(PlanarLinearPolygonMesh<N, T, I> const & mesh, Ray2<T> const & ray,
          T * const intersections, Size * const n) noexcept
{
  T const r_miss = infiniteDistance<T>();
  Size nintersect = 0;
#ifndef NDEBUG
  Size const n0 = *n;
#endif
  for (Size i = 0; i < numFaces(mesh); ++i) {
    auto const face = mesh.getFace(i);
    for (Size j = 0; j < polygonNumEdges<1, N>(); ++j) {
      auto const edge = face.getEdge(j);
      T const r = intersect(edge, ray);
      if (r < r_miss) {
        assert(nintersect < n0);
        intersections[nintersect++] = r;
      }
    }
  }
  *n = nintersect;
  std::sort(intersections, intersections + nintersect);
}

template <Size N, std::floating_point T, std::signed_integral I>
void
intersect(PlanarQuadraticPolygonMesh<N, T, I> const & mesh, Ray2<T> const & ray,
          T * const intersections, Size * const n) noexcept
{
  T const r_miss = infiniteDistance<T>();
  Size nintersect = 0;
#ifndef NDEBUG
  Size const n0 = *n;
#endif
  for (Size i = 0; i < numFaces(mesh); ++i) {
    auto const face = mesh.getFace(i);
    for (Size j = 0; j < polygonNumEdges<2, N>(); ++j) {
      auto const edge = face.getEdge(j);
      auto const r = intersect(edge, ray);
      if (r[0] < r_miss) {
        assert(nintersect < n0);
        intersections[nintersect++] = r[0];
      }
      if (r[1] < r_miss) {
        assert(nintersect < n0);
        intersections[nintersect++] = r[1];
      }
    }
  }
  *n = nintersect;
  std::sort(intersections, intersections + nintersect);
}

} // namespace um2
