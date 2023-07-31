#include <iostream>

namespace um2
{
// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr QuadraticQuadrilateral<D, T>::Polytope(
    Point<D, T> const & p0, Point<D, T> const & p1, Point<D, T> const & p2,
    Point<D, T> const & p3, Point<D, T> const & p4, Point<D, T> const & p5,
    Point<D, T> const & p6, Point<D, T> const & p7) noexcept
    : v{p0, p1, p2, p3, p4, p5, p6, p7}
{
}

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::operator()(R const r, S const s) const noexcept
    -> Point<D, T>
{
  T const xi = 2 * static_cast<T>(r) - 1;
  T const eta = 2 * static_cast<T>(s) - 1;
  T const w[8] = {(1 - xi) * (1 - eta) * (-xi - eta - 1) / 4,
                  (1 + xi) * (1 - eta) * (xi - eta - 1) / 4,
                  (1 + xi) * (1 + eta) * (xi + eta - 1) / 4,
                  (1 - xi) * (1 + eta) * (-xi + eta - 1) / 4,
                  (1 - xi * xi) * (1 - eta) / 2,
                  (1 - eta * eta) * (1 + xi) / 2,
                  (1 - xi * xi) * (1 + eta) / 2,
                  (1 - eta * eta) * (1 - xi) / 2};
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w[0] * v[0][i] + w[1] * v[1][i] + w[2] * v[2][i] + w[3] * v[3][i] +
                w[4] * v[4][i] + w[5] * v[5][i] + w[6] * v[6][i] + w[7] * v[7][i];
  }
  return result;
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::jacobian(R r, S s) const noexcept -> Mat<D, 2, T>
{
  T const xi = 2 * static_cast<T>(r) - 1;
  T const eta = 2 * static_cast<T>(s) - 1;
  T const xi_eta = xi * eta;
  T const xi_xi = xi * xi;
  T const eta_eta = eta * eta;
  T const w0 = (eta - eta_eta) / 2;
  T const w1 = (eta + eta_eta) / 2;
  T const w2 = (xi - xi_eta);
  T const w3 = (xi + xi_eta);
  T const w4 = 1 - eta_eta;
  T const w5 = (xi - xi_xi) / 2;
  T const w6 = (xi + xi_xi) / 2;
  T const w7 = eta - xi_eta;
  T const w8 = eta + xi_eta;
  T const w9 = 1 - xi_xi;
  Mat<D, 2, T> result;
  for (Size i = 0; i < D; ++i) {
    result.col(0)[i] = w0 * (v[0][i] - v[1][i]) + w1 * (v[2][i] - v[3][i]) +
                       w2 * (v[0][i] + v[1][i] - 2 * v[4][i]) +
                       w3 * (v[2][i] + v[3][i] - 2 * v[6][i]) + w4 * (v[5][i] - v[7][i]);
    result.col(1)[i] = w5 * (v[0][i] - v[3][i]) + w6 * (v[2][i] - v[1][i]) +
                       w7 * (v[0][i] + v[3][i] - 2 * v[7][i]) +
                       w8 * (v[1][i] + v[2][i] - 2 * v[5][i]) + w9 * (v[6][i] - v[4][i]);
  }
  return result;
}

// -------------------------------------------------------------------
// edge
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::edge(Size i) const noexcept -> QuadraticSegment<D, T>
{
  assert(i < 4);
  return (i == 3) ? QuadraticSegment<D, T>(v[3], v[0], v[7])
                  : QuadraticSegment<D, T>(v[i], v[i + 1], v[i + 4]);
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  static_assert(D == 2,
                "QuadraticQuadrilateral::contains() is only defined for 2D quads");
  for (Size i = 0; i < 4; ++i) {
    if (!edge(i).isLeft(p)) {
      return false;
    }
  }
  return true;
}

// -------------------------------------------------------------------
// linearPolygon
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::linearPolygon() const noexcept -> Quadrilateral<D, T>
{
  return Quadrilateral<D, T>(v[0], v[1], v[2], v[3]);
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::area() const noexcept -> T
{
  static_assert(D == 2, "QuadraticQuadrilateral::area() is only defined for 2D quads");
  T result = linearPolygon().area();
  for (Size i = 0; i < 4; ++i) {
    result += edge(i).enclosedArea();
  }
  return result;
}

// -------------------------------------------------------------------
// centroid
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::centroid() const noexcept -> Point<D, T>
{
  static_assert(D == 2,
                "QuadraticQuadrilateral::centroid() is only defined for 2D quads");
  // By geometric decomposition
  auto const quad = linearPolygon();
  T area_sum = quad.area();
  Point2<T> centroid_sum = area_sum * quad.centroid();
  for (Size i = 0; i < 4; ++i) {
    auto const e = this->edge(i);
    T const a = e.enclosedArea();
    area_sum += a;
    centroid_sum += a * e.enclosedCentroid();
  }
  return centroid_sum / area_sum;
}

// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  auto result = edge(0).boundingBox();
  for (Size i = 1; i < 4; ++i) {
    result = um2::boundingBox(result, edge(i).boundingBox());
  }
  return result;
}

} // namespace um2
