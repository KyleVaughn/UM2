// Free functions
#include <um2/geometry/dion/boundingBox.inl>    
#include <um2/geometry/dion/interpolate.inl>    
#include <um2/geometry/dion/jacobian.inl>      
#include <um2/geometry/dion/getRotation.inl>
#include <um2/geometry/dion/length.inl>    
#include <um2/geometry/dion/pointClosestTo.inl>    
#include <um2/geometry/dion/pointIsLeft.inl>
#include <um2/geometry/dion/QuadraticSegment.inl>

// Member functions
namespace um2
{

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
