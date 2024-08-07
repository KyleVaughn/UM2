#pragma once

#include <um2/config.hpp>
#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/stdlib/math/inverse_trigonometric_functions.hpp>
#include <um2/stdlib/math/rounding_functions.hpp>
#include <um2/stdlib/math/trigonometric_functions.hpp>
#include <um2/stdlib/numbers.hpp>

//==============================================================================
// MODULAR RAY PARAMETERS
//==============================================================================
// See the MPACT Theory Manual Version 4.3 (ORNL/SPR-2021/2330) for a detailed
// description of modular ray tracing.
//
// Note that for a target azimuthal angle γ ∈ (0, π) and ray spacing s, the
// axis-aligned box (ray tracing module) will likely not be the necessary size
// to produce cyclic rays at exactly the target angle or spacing. Instead, an
// effective angle and spacing are computed to ensure that the rays are cyclic.

namespace um2
{

// Modular ray parameters for a single angle
template <class T>
class ModularRayParams
{

  AxisAlignedBox2<T> _box;
  Vec2I _num_rays;    // Number of rays spawned on the box's x and y edges
  Vec2<T> _spacing;   // Spacing between rays in x and y
  Vec2<T> _direction; // Direction of rays

public:
  //============================================================================
  // Constructors
  //============================================================================

  constexpr ModularRayParams() noexcept = default;

  // a: Target azimuthal angle γ ∈ (0, π)
  // s: Target ray spacing
  HOSTDEV constexpr ModularRayParams(T a, T s, AxisAlignedBox2<T> box) noexcept;

  //============================================================================
  // Methods
  //============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getRay(Int i) const noexcept -> Ray2<T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getTotalNumRays() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getNumXRays() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getNumYRays() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getSpacing() const noexcept -> Vec2<T>;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  getDirection() const noexcept -> Vec2<T>;
};

//==============================================================================
// Constructors
//==============================================================================

// a: azimuthal angle γ ∈ (0, π)
// s: ray spacing
// w: width of ray tracing module
// h: height of ray tracing module
template <class T>
HOSTDEV constexpr ModularRayParams<T>::ModularRayParams(
    T const a, T const s, AxisAlignedBox2<T> const box) noexcept
    : _box(box)
{
  ASSERT_ASSUME(0 < a);
  ASSERT_ASSUME(a < um2::pi<T>);
  ASSERT_ASSUME(0 < s);

  // width and height of the box
  auto const wh = box.extents();

  // Number of rays in the x and y directions
  Vec2<T> const num_rays_t(um2::ceil(um2::abs(wh[0] * um2::sin(a) / s)),
                           um2::ceil(um2::abs(wh[1] * um2::cos(a) / s)));

  _num_rays[0] = static_cast<Int>(num_rays_t[0]);
  _num_rays[1] = static_cast<Int>(num_rays_t[1]);
  ASSERT(_num_rays[0] > 0);
  ASSERT(_num_rays[1] > 0);
  _spacing = wh / num_rays_t;

  // Effective angle to ensure cyclic rays
  auto const a_eff = um2::atan(_spacing[1] / _spacing[0]);
  _direction[0] = um2::cos(a_eff);
  if (a > pi_2<T>) {
    _direction[0] = -_direction[0];
  }
  _direction[1] = um2::sin(a_eff);
}

//==============================================================================
// Methods
//==============================================================================

template <class T>
HOSTDEV constexpr auto
ModularRayParams<T>::getRay(Int const i) const noexcept -> Ray2<T>
{
  // Angle < π/2
  //
  // i < nx (Case 0)
  // Moving from x_max to x_min along x-axis
  // x0 = x_max - (i + 0.5) * dx
  // y0 = y_min
  //
  // i >= nx (Case 1)
  // Moving from y_min to y_max along y-axis
  // x0 = x_min
  // y0 = y_min + (i - nx + 0.5) * dy
  //
  // Angle > π/2
  //
  // i < ny (Case 2)
  // Moving from x_min to x_max along x-axis
  // x0 = x_min + (i + 0.5) * dx
  // y0 = y_min
  //
  // i >= ny (Case 3)
  // Moving from y_min to y_max along y-axis
  // x0 = x_max
  // y0 = y_min + (i - ny + 0.5) * dy
  ASSERT_ASSUME(i < _num_rays[0] + _num_rays[1]);
  int case_id = (i < _num_rays[0]) ? 0 : 1;
  case_id += (_direction[0] < 0) ? 2 : 0;
  Point2<T> origin = _box.minima();
  auto const i_half = static_cast<T>(i) + static_cast<T>(1) / 2;
  switch (case_id) {
  case 0:
    origin[0] = _box.maxima(0) - _spacing[0] * i_half;
    break;
  case 1:
    origin[1] += _spacing[1] * (i_half - static_cast<T>(_num_rays[0]));
    break;
  case 2:
    origin[0] += _spacing[0] * i_half;
    break;
  case 3:
    origin[0] = _box.maxima(0);
    origin[1] += _spacing[1] * (i_half - static_cast<T>(_num_rays[0]));
    break;
  default:
    __builtin_unreachable();
  }
  Ray2<T> res(origin, _direction);
  return res;
}

template <class T>
HOSTDEV constexpr auto
ModularRayParams<T>::getTotalNumRays() const noexcept -> Int
{
  return _num_rays[0] + _num_rays[1];
}

template <class T>
HOSTDEV constexpr auto
ModularRayParams<T>::getNumXRays() const noexcept -> Int
{
  return _num_rays[0];
}

template <class T>
HOSTDEV constexpr auto
ModularRayParams<T>::getNumYRays() const noexcept -> Int
{
  return _num_rays[1];
}

template <class T>
HOSTDEV constexpr auto
ModularRayParams<T>::getSpacing() const noexcept -> Vec2<T>
{
  return _spacing;
}

template <class T>
HOSTDEV constexpr auto
ModularRayParams<T>::getDirection() const noexcept -> Vec2<T>
{
  return _direction;
}

} // namespace um2
