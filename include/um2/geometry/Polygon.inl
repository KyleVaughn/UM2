namespace um2
{

// -------------------------------------------------------------------
// Polygon (K = 2)
// -------------------------------------------------------------------
// For Polygons
//   Triangle (P = 1, N = 3)
//   Quadrilateral (P = 1, N = 4)
//   Quadratic Triangle (P = 2, N = 6)
//   Quadratic Quadrilateral (P = 2, N = 8)
// Defines:
//   edge
//   area
//   centroid
//   contains (point)
//   boundingBox

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
edge(LinearPolygon<N, D, T> const & p, Size const i) noexcept
{
  assert(0 <= i && i < N);
  return (i < N - 1) ? LineSegment<D, T>(p[i], p[i + 1])
                     : LineSegment<D, T>(p[N - 1], p[0]);
}

template <Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
edge(QuadraticPolygon<N, D, T> const & p, Size const i) noexcept
{
  assert(0 <= i && i < N);
  constexpr Size m = N / 2;
  return (i < m - 1) ? QuadraticSegment<D, T>(p[i], p[i + 1], p[i + m])
                     : QuadraticSegment<D, T>(p[m - 1], p[0], p[N - 1]);
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
area(Triangle<D, T> const & tri) noexcept -> T
{
  Vec<D, T> const v10 = tri[1] - tri[0];
  Vec<D, T> const v20 = tri[2] - tri[0];
  if constexpr (D == 2) {
    return v10.cross(v20) / 2; // this is the signed area
  } else if constexpr (D == 3) {
    return v10.cross(v20).norm() / 2; // this is the unsigned area
  } else {
    static_assert(D == 2 || D == 3,
                  "Triangle::area() is only defined for 2D and 3D triangles");
  }
}

// Area of a planar linear polygon
// template <Size N, Size D, typename T>
// PURE HOSTDEV constexpr
// T area(Polytope<K, P, N, D, T> const & poly)
//{
//    if constexpr (N == 3) {
//        return triangle_area(poly[0], poly[1], poly[2]);
//    } else if constexpr (N == 4) {
//        return quadrilateral_area(poly[0], poly[1], poly[2], poly[3]);
//    } else {
//        // Shoelace forumla A = 1/2 * sum_{i=0}^{n-1} cross(p_i, p_{i+1})
//        // p_n = p_0
//        T sum = cross(poly[N - 1], poly[0]); // cross(p_{n-1}, p_0), the last term
//        for (Size i = 0; i < N - 1; ++i) {
//            sum += cross(poly[i], poly[i + 1]);
//        }
//        return sum / 2;
//    }
//}
//
// template <Size D, typename T>
// PURE HOSTDEV static constexpr
// Triangle<D, T> linear_polygon(QuadraticTriangle<D, T> const & tri)
//{
//    return {tri[0], tri[1], tri[2]};
//}
//
// template <Size D, typename T>
// PURE HOSTDEV static constexpr
// Quadrilateral<D, T> linear_polygon(QuadraticQuadrilateral<D, T> const & quad)
//{
//    return {quad[0], quad[1], quad[2], quad[3]};
//}
//
//// Area of a quadratic, planar polygon
// template <Size K, Size P, Size N, Size D, typename T>
// requires (K == 2 && P == 2 && D == 2)
// PURE HOSTDEV constexpr
// T area(Polytope<K, P, N, D, T> const & poly)
//{
//     T result = area(linear_polygon(poly));
//     for (Size i = 0; i < N / 2; ++i) {
//         result += enclosed_area(poly.edge(i));
//     }
//     return result;
// }
//
//// -------------------------------------------------------------------
//// centroid
//// -------------------------------------------------------------------
//
// template <Size D, typename T>
// PURE HOSTDEV constexpr
// Point<D, T> triangle_centroid(Point<D, T> const & p0,
//                              Point<D, T> const & p1,
//                              Point<D, T> const & p2)
//{
//    return static_cast<T>(0.3333333333333333) * (p0 + p1 + p2);
//}
//
// template <Size D, typename T>
// PURE HOSTDEV constexpr
// Point2<T> quadrilateral_centroid(Point<D, T> const & p0,
//                                 Point<D, T> const & p1,
//                                 Point<D, T> const & p2,
//                                 Point<D, T> const & p3)
//{
//    Vec2<T> const v01 = p1 - p0;
//    Vec2<T> const v02 = p2 - p0;
//    Vec2<T> const v03 = p3 - p0;
//    T const a1 = cross(v01, v02);
//    T const a2 = cross(v02, v03);
//    return (a1 * (p0 + p1 + p2) +
//            a2 * (p0 + p2 + p3)) / (3 * (a1 + a2));
//}
//
// template <typename T>
// PURE HOSTDEV constexpr
// Point3<T> centroid(Triangle<3, T> const & tri) {
//    return triangle_centroid(tri[0], tri[1], tri[2]);
//}
//
//// Centroid of a linear, planar polygon
// template <Size K, Size P, Size N, Size D, typename T>
// requires (K == 2 && P == 1 && D == 2)
// PURE HOSTDEV constexpr
// Point2<T> centroid(Polytope<K, P, N, D, T> const & poly)
//{
//     if constexpr (N == 3) {
//         return triangle_centroid(poly[0], poly[1], poly[2]);
//     } else if constexpr (N == 4) {
//         return quadrilateral_centroid(poly[0], poly[1], poly[2], poly[3]);
//     } else {
//         // Similar to the shoelace formula.
//         // C = 1/6A * sum_{i=0}^{n-1} cross(p_i, p_{i+1}) * (p_i + p_{i+1})
//         T area_sum = cross(poly[N - 1], poly[0]); // p_{n-1} x p_0, the last term
//         Point2<T> centroid_sum = area_sum * (poly[N - 1] + poly[0]);
//         for (Size i = 0; i < N - 1; ++i) {
//             T const area = cross(poly[i], poly[i + 1]);
//             area_sum += area;
//             centroid_sum += area * (poly[i] + poly[i + 1]);
//         }
//         return centroid_sum / (3 * area_sum);
//     }
// }
//
//// Centroid of a quadratic, planar polygon
// template <Size K, Size P, Size N, Size D, typename T>
// requires (K == 2 && P == 2 && D == 2)
// PURE HOSTDEV constexpr
// Point2<T> centroid(Polytope<K, P, N, D, T> const & poly)
//{
//     // By geometric decomposition
//     auto const & lin_poly = linear_polygon(poly);
//     T area_sum = area(lin_poly);
//     Point2<T> centroid_sum = area_sum * centroid(lin_poly);
//     for (Size i = 0; i < N / 2; ++i) {
//         auto const & edge = poly.edge(i);
//         T const area = enclosed_area(edge);
//         area_sum += area;
//         centroid_sum += area * enclosed_area_centroid(edge);
//     }
//     return centroid_sum / area_sum;
// }
//
//// -------------------------------------------------------------------
//// contains
//// -------------------------------------------------------------------
//
// template <Size K, Size P, Size N, Size D, typename T>
// PURE HOSTDEV constexpr
// bool Polytope<K, P, N, D, T>:: contains(Point<D, T> const & p) const
// requires (K == 2 && D == 2)
//{
//    if constexpr (P == 1) {
//        // Linear polygon
//        for (Size i = 0; i < N; ++i) {
//            if (!this->edge(i).point_is_left(p)) {
//                return false;
//            }
//        }
//        return true;
//    } else if constexpr (P == 2) {
//        // Quadratic polygon
//        for (Size i = 0; i < N / 2; ++i) {
//            if (!this->edge(i).point_is_left(p)) {
//                return false;
//            }
//        }
//        return true;
//    } else {
//        static_assert(!K, "Unsupported polytope");
//    }
//}
//
} // namespace um2
