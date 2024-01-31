#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/math/angular_quadrature.hpp>
#include <um2/stdlib/math.hpp>

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
class ModularRayParams
{

  AxisAlignedBox2 _box;
  Vec2<I> _num_rays;  // Number of rays spawned on the box's x and y edges
  Vec2<F> _spacing;   // Spacing between rays in x and y
  Vec2<F> _direction; // Direction of rays

public:
  //============================================================================
  // Constructors
  //============================================================================

  constexpr ModularRayParams() noexcept = default;

  // a: Target azimuthal angle γ ∈ (0, π)
  // s: Target ray spacing
  HOSTDEV constexpr ModularRayParams(F a, F s, AxisAlignedBox2 box) noexcept;

  //============================================================================
  // Methods
  //============================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  getRay(I i) const noexcept -> Ray2;

  HOSTDEV [[nodiscard]] constexpr auto
  getTotalNumRays() const noexcept -> I;

  HOSTDEV [[nodiscard]] constexpr auto
  getNumXRays() const noexcept -> I;

  HOSTDEV [[nodiscard]] constexpr auto
  getNumYRays() const noexcept -> I;

  HOSTDEV [[nodiscard]] constexpr auto
  getSpacing() const noexcept -> Vec2<F>;

  HOSTDEV [[nodiscard]] constexpr auto
  getDirection() const noexcept -> Vec2<F>;
};

//==============================================================================
// Constructors
//==============================================================================

// a: azimuthal angle γ ∈ (0, π)
// s: ray spacing
// w: width of ray tracing module
// h: height of ray tracing module
HOSTDEV constexpr ModularRayParams::ModularRayParams(F const a, F const s,
                                                     AxisAlignedBox2 const box) noexcept
    : _box(box)
{
  ASSERT_ASSUME(0 < a);
  ASSERT_ASSUME(a < pi<F>);
  ASSERT_ASSUME(0 < s);

  auto const w = box.width();
  auto const h = box.height();

  // Number of rays in the x and y directions
  Vec2<F> const num_rays_t(um2::ceil(um2::abs(w * um2::sin(a) / s)),
                           um2::ceil(um2::abs(h * um2::cos(a) / s)));

  _num_rays[0] = static_cast<I>(num_rays_t[0]);
  _num_rays[1] = static_cast<I>(num_rays_t[1]);
  _spacing[0] = w / num_rays_t[0];
  _spacing[1] = h / num_rays_t[1];

  // Effective angle to ensure cyclic rays
  auto const a_eff = um2::atan(_spacing[1] / _spacing[0]);
  _direction[0] = um2::cos(a_eff);
  if (a > pi_2<F>) {
    _direction[0] = -_direction[0];
  }
  _direction[1] = um2::sin(a_eff);
}

//==============================================================================
// Methods
//==============================================================================

HOSTDEV constexpr auto
ModularRayParams::getRay(I const i) const noexcept -> Ray2
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
  Point2 origin = _box.minima();
  auto const i_half = static_cast<F>(i) + static_cast<F>(1) / 2;
  switch (case_id) {
  case 0:
    origin[0] = _box.maxima(0) - _spacing[0] * i_half;
    break;
  case 1:
    origin[1] += _spacing[1] * (i_half - static_cast<F>(_num_rays[0]));
    break;
  case 2:
    origin[0] += _spacing[0] * i_half;
    break;
  case 3:
    origin[0] = _box.maxima(0);
    origin[1] += _spacing[1] * (i_half - static_cast<F>(_num_rays[0]));
    break;
  default:
    __builtin_unreachable();
  }
  Ray2 res(origin, _direction);
  return res;
}

HOSTDEV constexpr auto
ModularRayParams::getTotalNumRays() const noexcept -> I
{
  return _num_rays[0] + _num_rays[1];
}

HOSTDEV constexpr auto
ModularRayParams::getNumXRays() const noexcept -> I
{
  return _num_rays[0];
}

HOSTDEV constexpr auto
ModularRayParams::getNumYRays() const noexcept -> I
{
  return _num_rays[1];
}

HOSTDEV constexpr auto
ModularRayParams::getSpacing() const noexcept -> Vec2<F>
{
  return _spacing;
}

HOSTDEV constexpr auto
ModularRayParams::getDirection() const noexcept -> Vec2<F>
{
  return _direction;
}

} // namespace um2
