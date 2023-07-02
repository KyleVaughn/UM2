#pragma once

#include <um2/geometry/LineSegment.hpp>
#include <um2/ray_casting/Ray.hpp>

namespace um2
{
    
// Returns the value r such that R(r) = L(s). 
// If such a value does not exist, infiniteDistance<T> is returned instead.    
// 1) P₁ + s(P₂ - P₁) = O + r𝗱           subtracting P₁ from both sides    
// 2) s(P₂ - P₁) = (O - P₁) + r𝗱         let 𝘂 = O - P₁, 𝘃 = P₂-P₁    
// 3) s𝘃 = 𝘂 + r𝗱                        cross product with 𝗱 (distributive)    
// 4) s(𝘃 × 𝗱) = 𝘂 × 𝗱  + r(𝗱 × 𝗱)       𝗱 × 𝗱 = 𝟬       
// 5) s(𝘃 × 𝗱) = 𝘂 × 𝗱                   let 𝘃 × 𝗱 = 𝘇 and 𝘂 × 𝗱 = 𝘅    
// 6) s𝘇 = 𝘅                             dot product 𝘇 to each side    
// 7) s𝘇 ⋅ 𝘇 = 𝘅 ⋅ 𝘇                     divide by 𝘇 ⋅ 𝘇    
// 8) s = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)    
// If s ∉ [0, 1] the intersections is invalid. If s ∈ [0, 1],    
// 1) O + r𝗱 = P₁ + s𝘃                   subtracting O from both sides        
// 2) r𝗱 = -𝘂 + s𝘃                       cross product with 𝘃        
// 3) r(𝗱 × 𝘃) = -𝘂 × 𝘃 + s(𝘃 × 𝘃)       𝘃  × 𝘃 = 𝟬         
// 4) r(𝗱 × 𝘃) = -𝘂 × 𝘃                  using 𝗱 × 𝘃 = -(𝘃 × 𝗱)    
// 5) r(𝘃 × 𝗱) = 𝘂 × 𝘃                   let 𝘂 × 𝘃 = 𝘆    
// 6) r𝘇 = 𝘆                             dot product 𝘇 to each side        
// 7) r(𝘇 ⋅ 𝘇) = 𝘆 ⋅ 𝘇                   divide by (𝘇 ⋅ 𝘇)    
// 9) r = (𝘆 ⋅ 𝘇)/(𝘇 ⋅ 𝘇)    
//    
// The cross product of two vectors in the plane is a vector of the form (0, 0, k),        
// hence, in 2D:        
// s = (𝘅 ⋅ 𝘇)/(𝘇 ⋅ 𝘇) = x₃/z₃     
// r = (𝘆 ⋅ 𝘇)/(𝘇 ⋅ 𝘇) = y₃/z₃     
// This result is valid if s ∈ [0, 1]   
template <std::floating_point T>
PURE constexpr auto 
intersect(Ray2<T> const & ray, LineSegment2<T> const & line) noexcept -> T
{
  Vec2<T> const v = line[1] - line[0];
  Vec2<T> const u = ray.o - line[0];

  T const z = cross2(v, ray.d);

  T const s = cross2(u, ray.d) / z;
  T const r = cross2(u, v) / z;

  bool const valid = 0 <= s && s <= 1;

  return valid ? r : infiniteDistance<T>();
}

} // namespace um2
