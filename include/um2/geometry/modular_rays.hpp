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
  Vec2<Size> num_rays; // Number of rays in x and y
  Vec2<T> spacing;     // Spacing between rays in x and y
  Vec2<T> origin;      // Origin of first ray
  Vec2<T> direction;   // Direction of rays
}

// a: azimuthal angle γ ∈ (0, π/2)
// s: ray spacing
// w: width of ray tracing module
// h: height of ray tracing module
template <typename T>
HOSTDEV constexpr auto
getModularRayParams(T const a, T const s, T const w, T const h) -> ModularRayParams<T>
{
  assert(0 < a && a < pi<T> / 2);
  assert(0 < s);
  // Number of rays in the x and y directions
  Vec2<T> const num_rays_t(um2::ceil(um2::abs(w * um2::sin(a) / s)),
                           um2::ceil(um2::abs(h * um2::cos(a) / s)));
  ModularRayParams<T> res;
  res.num_rays[0] = static_cast<Size>(num_rays_t[0]);
  res.num_rays[1] = static_cast<Size>(num_rays_t[1]);

  // Effective angle to ensure cyclic rays
  T const a_eff = um2::atan((h * nx_t) / (w * ny_t));
  res.direction[0] = um2::cos(a_eff);
  res.direction[1] = um2::sin(a_eff);

  res.spacing[0] = -w / nx_t;
  res.spacing[1] = h / ny_t;

  res.origin[0] = w - res.spacing[0] / 2;
  res.origin[1] = h - res.spacing[1] / 2;

  return res;
}

} // namespace um2
