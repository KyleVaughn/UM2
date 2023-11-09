#pragma once

#include <um2/geometry/axis_aligned_box.hpp>
#include <um2/geometry/ray.hpp>
#include <um2/math/angular_quadrature.hpp>
#include <um2/stdlib/math.hpp>

namespace um2
{

// Modular ray parameters for a single angle
template <typename T>
struct ModularRayParams {

  AxisAlignedBox2<T> box;
  Vec2<Size> num_rays; // Number of rays in x and y
  Vec2<T> spacing;     // Spacing between rays in x and y
  Vec2<T> direction;   // Direction of rays

  HOSTDEV [[nodiscard]] constexpr auto
  getRay(Size i) const noexcept -> Ray2<T>
  {
    // Angle < π/2
    //
    // i < nx (Case 0)
    // x0 = x_max - (i + 0.5) * dx
    // y0 = y_min
    //
    // i >= nx (Case 1)
    // x0 = x_min
    // y0 = y_min + (i - nx + 0.5) * dy
    //
    // Angle > π/2
    //
    // i < ny (Case 2)
    // x0 = x_min + (i + 0.5) * dx
    // y0 = y_min
    //
    // i >= ny (Case 3)
    // x0 = x_max
    // y0 = y_min + (i - ny + 0.5) * dy
    ASSERT_ASSUME(i < num_rays[0] + num_rays[1]);
    int case_id = i < num_rays[0] ? 0 : 1;
    case_id += direction[0] < 0 ? 2 : 0;
    Ray2<T> res(box.minima, direction);
    switch (case_id) {
    case 0:
      res.o[0] = box.maxima[0] - spacing[0] * (static_cast<T>(i) + static_cast<T>(0.5));
      break;
    case 1:
      res.o[1] += spacing[1] * (static_cast<T>(i - num_rays[0]) + static_cast<T>(0.5));
      break;
    case 2:
      res.o[0] += spacing[0] * (static_cast<T>(i) + static_cast<T>(0.5));
      break;
    case 3:
      res.o[0] = box.maxima[0];
      res.o[1] += spacing[1] * (static_cast<T>(i - num_rays[0]) + static_cast<T>(0.5));
      break;
    default:
      __builtin_unreachable();
    }
    return res;
  }
};

// a: azimuthal angle γ ∈ (0, π)
// s: ray spacing
// w: width of ray tracing module
// h: height of ray tracing module
template <typename T>
HOSTDEV constexpr auto
getModularRayParams(T const a, T const s, AxisAlignedBox2<T> const box)
    -> ModularRayParams<T>
{
  ASSERT_ASSUME(0 < a);
  ASSERT_ASSUME(a < pi<T>);
  ASSERT_ASSUME(0 < s);

  T const w = box.width();
  T const h = box.height();

  // Number of rays in the x and y directions
  Vec2<T> const num_rays_t(um2::ceil(um2::abs(w * um2::sin(a) / s)),
                           um2::ceil(um2::abs(h * um2::cos(a) / s)));

  ModularRayParams<T> res;
  res.box = box;
  res.num_rays[0] = static_cast<Size>(num_rays_t[0]);
  res.num_rays[1] = static_cast<Size>(num_rays_t[1]);
  res.spacing[0] = w / num_rays_t[0];
  res.spacing[1] = h / num_rays_t[1];

  // Effective angle to ensure cyclic rays
  T const a_eff = um2::atan(res.spacing[1] / res.spacing[0]);
  res.direction[0] = um2::cos(a_eff);
  if (a > pi<T> / 2) {
    res.direction[0] *= -1;
  }
  res.direction[1] = um2::sin(a_eff);

  return res;
}

} // namespace um2
