#pragma once

#include <um2/geometry/AxisAlignedBox.hpp>
#include <um2/geometry/Ray.hpp>
#include <um2/math/AngularQuadrature.hpp>
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
};

// a: azimuthal angle γ ∈ (0, π/2)
// s: ray spacing
// w: width of ray tracing module
// h: height of ray tracing module
template <typename T>
HOSTDEV constexpr auto
getModularRayParams(T const a, T const s, AxisAlignedBox2<T> const box) -> ModularRayParams<T>
{
  assert(0 < a && a < pi<T>);
  assert(0 < s);

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
