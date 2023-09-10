// Free functions
#include <um2/geometry/dion/QuadraticSegment.inl>
#include <um2/geometry/dion/boundingBox.inl>
#include <um2/geometry/dion/getRotation.inl>
#include <um2/geometry/dion/intersect.inl>
#include <um2/geometry/dion/length.inl>
#include <um2/geometry/dion/pointClosestTo.inl>
#include <um2/geometry/dion/pointIsLeft.inl>

// Member functions
namespace um2
{

//==============================================================================
//==============================================================================
// Free functions
//==============================================================================
//==============================================================================

//==============================================================================
// Interpolation
//==============================================================================

template <Size D, typename T, typename R>
PURE HOSTDEV constexpr auto
interpolate(LineSegment<D, T> const & l, R const r) noexcept -> Point<D, T>
{
  T const rr = static_cast<T>(r);
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = l[0][i] + rr * (l[1][i] - l[0][i]);
  }
  return result;
}

template <Size D, typename T, typename R>
PURE HOSTDEV constexpr auto
interpolate(QuadraticSegment<D, T> const & q, R const r) noexcept -> Point<D, T>
{
  // (2 * r - 1) * (r - 1) * v0 +
  // (2 * r - 1) *  r      * v1 +
  // -4 * r      * (r - 1) * v2
  T const rr = static_cast<T>(r);
  T const two_rr_1 = 2 * rr - 1;
  T const rr_1 = rr - 1;

  T const w0 = two_rr_1 * rr_1;
  T const w1 = two_rr_1 * rr;
  T const w2 = -4 * rr * rr_1;
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] = w0 * q[0][i] + w1 * q[1][i] + w2 * q[2][i];
  }
  return result;
} 

//==============================================================================
// jacobian
//==============================================================================

template <Size D, typename T, typename R>    
PURE HOSTDEV constexpr auto    
jacobian(LineSegment<D, T> const & l, R const /*r*/) noexcept -> Point<D, T>    
{    
  return l[1] - l[0];    
}    
    
template <Size D, typename T, typename R>    
PURE HOSTDEV constexpr auto    
jacobian(QuadraticSegment<D, T> const & q, R const r) noexcept -> Point<D, T>    
{    
  // (4 * r - 3) * (v0 - v2) + (4 * r - 1) * (v1 - v2)    
  T const w0 = 4 * static_cast<T>(r) - 3;    
  T const w1 = 4 * static_cast<T>(r) - 1;    
  Vec<D, T> result;    
  for (Size i = 0; i < D; ++i) {    
    result[i] = w0 * (q[0][i] - q[2][i]) + w1 * (q[1][i] - q[2][i]);    
  }    
  return result;    
}

//==============================================================================
//==============================================================================
// Member functions 
//==============================================================================
//==============================================================================

//==============================================================================
// Accessors
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return v[i];
}

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return v[i];
}

//==============================================================================
// Interpolation
//==============================================================================

template <Size P, Size N, Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  return interpolate(*this, r);
}

//==============================================================================
// jacobian
//==============================================================================

template <Size P, Size N, Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::jacobian(R const r) const noexcept -> Vec<D, T>
{
  return um2::jacobian(*this, r);
}

//==============================================================================
// getRotation
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::getRotation() const noexcept -> Mat<D, D, T>
{
  return um2::getRotation(*this);
}

//==============================================================================
// isLeft
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool
{
  return pointIsLeft(*this, p);
}

//==============================================================================
// length
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::length() const noexcept -> T
{
  return um2::length(*this);
}

//==============================================================================
// boundingBox
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  return um2::boundingBox(*this);
}

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::pointClosestTo(Point<D, T> const & p) const noexcept -> T
{
  return um2::pointClosestTo(*this, p);
}

//==============================================================================
// distanceTo
//==============================================================================

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::squaredDistanceTo(Point<D, T> const & p) const noexcept -> T
{
  T const r = pointClosestTo(p);
  Point<D, T> const p_closest = (*this)(r);
  return p_closest.squaredDistanceTo(p);
}

template <Size P, Size N, Size D, typename T>
PURE HOSTDEV constexpr auto
Dion<P, N, D, T>::distanceTo(Point<D, T> const & p) const noexcept -> T
{
  return um2::sqrt(squaredDistanceTo(p));
}

} // namespace um2
