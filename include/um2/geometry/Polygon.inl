// Free functions
#include <um2/geometry/polygon/area.inl>
#include <um2/geometry/polygon/boundingBox.inl>
#include <um2/geometry/polygon/centroid.inl>
#include <um2/geometry/polygon/contains.inl>
#include <um2/geometry/polygon/flipFace.inl>
#include <um2/geometry/polygon/getEdge.inl>
#include <um2/geometry/polygon/interpolate.inl>
#include <um2/geometry/polygon/isCCW.inl>
#include <um2/geometry/polygon/isConvex.inl>
#include <um2/geometry/polygon/jacobian.inl>
#include <um2/geometry/polygon/linearPolygon.inl>

// Member functions
namespace um2
{

//==============================================================================
// Accessors
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size P, Size N, Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::operator()(R const r, S const s) const noexcept -> Point<D, T>
{
  return interpolate(*this, r, s);
}

//==============================================================================
// jacobian
//==============================================================================

template <Size P, Size N, Size D, typename T>
template <typename R, typename S>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::jacobian(R r, S s) const noexcept -> Mat<D, 2, T>
{
  return um2::jacobian(*this, r, s);
}

//==============================================================================
// edge
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::getEdge(Size i) const noexcept -> Edge
{
  return um2::getEdge(*this, i);
}

//==============================================================================
// contains
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::contains(Point<D, T> const & p) const noexcept -> bool
{
  return um2::contains(*this, p);
}

//==============================================================================
// area
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::area() const noexcept -> T
{
  return um2::area(*this);
}

//==============================================================================
// centroid
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::centroid() const noexcept -> Point<D, T>
{
  return um2::centroid(*this);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

//==============================================================================
// isCCW
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Polygon<P, N, D, T>::isCCW() const noexcept -> bool
{
  return um2::isCCW(*this);
}

} // namespace um2
