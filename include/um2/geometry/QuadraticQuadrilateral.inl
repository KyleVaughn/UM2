#include <iostream>

namespace um2
{

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
QuadraticQuadrilateral<D, T>::operator()(R const r, S const s) const noexcept -> Point<D, T>
{
  T const xi =  2 * static_cast<T>(r) - 1;    
  T const eta = 2 * static_cast<T>(s) - 1;    
  T const w[8] = {(1 - xi) * (1 - eta) * (-xi - eta - 1) / 4,    
                  (1 + xi) * (1 - eta) * ( xi - eta - 1) / 4,    
                  (1 + xi) * (1 + eta) * ( xi + eta - 1) / 4,    
                  (1 - xi) * (1 + eta) * (-xi + eta - 1) / 4,    
                             (1 -  xi *  xi) * (1 - eta) / 2,    
                             (1 - eta * eta) * (1 +  xi) / 2,    
                             (1 -  xi *  xi) * (1 + eta) / 2,    
                             (1 - eta * eta) * (1 -  xi) / 2};    
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
  T const rr = static_cast<T>(4 * r);
  T const ss = static_cast<T>(4 * s);
  T const tt = rr + ss - 3;
  Mat<D, 2, T> result;
  for (Size i = 0; i < D; ++i) {
    result.col(0)[i] = tt * (v[0][i] - v[3][i]) + (rr - 1) * (v[1][i] - v[3][i]) +
                       ss * (v[4][i] - v[5][i]);
    result.col(1)[i] = tt * (v[0][i] - v[5][i]) + (ss - 1) * (v[2][i] - v[5][i]) +
                       rr * (v[4][i] - v[3][i]);
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
  assert(i < 3);
  return (i == 2) ? QuadraticSegment<D, T>(v[2], v[0], v[5])
                  : QuadraticSegment<D, T>(v[i], v[i + 1], v[i + 3]);
}

// -------------------------------------------------------------------
// contains
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  static_assert(D == 2, "QuadraticQuadrilateral::contains() is only defined for 2D triangles");
  return edge(0).isLeft(p) && edge(1).isLeft(p) && edge(2).isLeft(p);
}

// -------------------------------------------------------------------
// linearPolygon
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::linearPolygon() const noexcept -> Quadrilateral<D, T>
{
  return Quadrilateral<D, T>(v[0], v[1], v[2]);
}

// -------------------------------------------------------------------
// area
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticQuadrilateral<D, T>::area() const noexcept -> T
{
  static_assert(D == 2, "QuadraticQuadrilateral::area() is only defined for 2D triangles");
  T result = linearPolygon().area();
  for (Size i = 0; i < 3; ++i) {
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
  static_assert(D == 2, "QuadraticQuadrilateral::centroid() is only defined for 2D triangles");
  // By geometric decomposition
  auto const tri = linearPolygon();
  T area_sum = tri.area();
  Point2<T> centroid_sum = area_sum * tri.centroid();
  for (Size i = 0; i < 3; ++i) {
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
  result = um2::boundingBox(result, edge(1).boundingBox());
  result = um2::boundingBox(result, edge(2).boundingBox());
  return result;
}

} // namespace um2
