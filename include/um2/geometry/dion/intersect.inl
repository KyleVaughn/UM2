namespace um2
{

//==============================================================================
// LineSegment
//==============================================================================

// Returns the value r such that R(r) = L(s).    
// If such a value does not exist, infiniteDistance<T> is returned instead.    
// 1) P₁ + s(P₂ - P₁) = O + rD           subtracting P₁ from both sides    
// 2) s(P₂ - P₁) = (O - P₁) + rD         let U = O - P₁, V = P₂-P₁    
// 3) sV = U + rD                        cross product with D (distributive)    
// 4) s(V × D) = U × D  + r(D × D)       D × D = 0    
// 5) s(V × D) = U × D                   let V × D = Z and U × D = X    
// 6) sZ = X                             dot product 𝘇 to each side    
// 7) sZ ⋅ Z = X ⋅ Z                     divide by Z ⋅ Z    
// 8) s = (X ⋅ Z)/(Z ⋅ Z)    
// If s ∉ [0, 1] the intersections is invalid. If s ∈ [0, 1],    
// 1) O + rD = P₁ + sV                   subtracting O from both sides    
// 2) rD = -U + sV                       cross product with 𝘃    
// 3) r(D × V) = -U × V + s(V × V)       V × V = 0    
// 4) r(D × V) = -U × V                  using D × V = -(V × D)    
// 5) r(V × D) = U × V                   let U × V = Y    
// 6) rZ = Y                             dot product Z to each side    
// 7) r(Z ⋅ Z) = Y ⋅ Z                   divide by (Z ⋅ Z)    
// 9) r = (Y ⋅ Z)/(Z ⋅ Z)    
//    
// The cross product of two vectors in the plane is a vector of the form (0, 0, k),    
// hence, in 2D:    
// s = (X ⋅ Z)/(Z ⋅ Z) = x₃/z₃    
// r = (Y ⋅ Z)/(Z ⋅ Z) = y₃/z₃    
// This result is valid if s ∈ [0, 1] 
template <typename T>
PURE HOSTDEV constexpr auto
intersect(LineSegment2<T> const & line, Ray2<T> const & ray) noexcept -> T
{
  Vec2<T> const v(line[1][0] - line[0][0], line[1][1] - line[0][1]);    
  Vec2<T> const u(ray.o[0] - line[0][0], ray.o[1] - line[0][1]);    
    
  T const z = v.cross(ray.d);    
    
  T const s = u.cross(ray.d) / z;    
  T r = u.cross(v) / z;    
    
  if (s < 0 || 1 < s) {    
    r = infiniteDistance<T>();    
  }
  return r;
}

//==============================================================================
// QuadraticSegment
//==============================================================================

// The ray: R(r) = O + rD
// The quadratic segment: Q(s) = C + sB + s²A,    
// where
//  C = P₁ 
//  B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]    
//  A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]    
// and    
// V₁₃ = q[3] - q[1]    
// V₂₃ = q[3] - q[2] 
//
// O + rD = C + sB + s²A                          subtracting C from both sides 
// rD = (C - O) + sB + s²A                        cross product with D (distributive)
// 0 = (C - O) × D + s(B × D) + s²(A × D)
// The cross product of two vectors in the plane is a vector of the form (0, 0, k).
// Let a = (A × D)ₖ, b = (B × D)ₖ, and c = ((C - O) × D)ₖ
// 0 = as² + bs + c
// If a = 0 
//   s = -c/b
// else
//   s = (-b ± √(b²-4ac))/2a
// s is invalid if b² < 4ac
// Once we have a valid s
// O + r𝗱 = P ⟹   r = ((P - O) ⋅ D)/(D ⋅ D)

template <typename T>
PURE HOSTDEV constexpr auto
intersect(QuadraticSegment2<T> const & q, Ray2<T> const & ray) noexcept -> Vec2<T>
{    
  // This code is called very frequently so we sacrifice readability for speed.
  Vec2<T> const v01(q[1][0] - q[0][0], q[1][1] - q[0][1]);
  Vec2<T> const v02(q[2][0] - q[0][0], q[2][1] - q[0][1]);
  Vec2<T> const v12(q[2][0] - q[1][0], q[2][1] - q[1][1]);

  Vec2<T> const A(-2 * (v02[0] + v12[0]), -2 * (v02[1] + v12[1]));
  Vec2<T> const B(3 * v02[0] + v12[0], 3 * v02[1] + v12[1]);
  // Vec2<T> const C = q[0];
  
  // Vec2<T> const D = ray.d;    
  // Vec2<T> const O = ray.o;    
  
  Vec2<T> const voc(q[0][0] - ray.o[0], q[0][1] - ray.o[1]);

  T const a = A.cross(ray.d);    
  T const b = B.cross(ray.d);    
  T const c = voc.cross(ray.d);    
  
  Vec2<T> result(infiniteDistance<T>(), infiniteDistance<T>());

  if (um2::abs(a) < static_cast<T>(1e-8)) {
    T const s = -c / b;
    if (0 <= s && s <= 1) {    
      Point2<T> const P(s * (s * A[0] + B[0]) + voc[0], 
                        s * (s * A[1] + B[1]) + voc[1]);    
      result[0] = dot(P, D) / D.squaredNorm(); 
    }    
    return result; 
  }    
  T const disc = b * b - 4 * a * c;
  if (disc < 0) {    
    return result; 
  }

  T const s1 = (-b - um2::sqrt(disc)) / (2 * a);    
  T const s2 = (-b + um2::sqrt(disc)) / (2 * a);    
  if (0 <= s1 && s1 <= 1)  {    
      Point2<T> const P(s1 * (s1 * A[0] + B[0]) + voc[0], 
                        s1 * (s1 * A[1] + B[1]) + voc[1]); 
      result[0] = dot(P, D) / D.squaredNorm(); 
  }    
  if (0 <= s2 && s2 <= 1)
  {    
    Point2<T> const P(s2 * (s2 * A[0] + B[0]) + voc[0], 
                      s2 * (s2 * A[1] + B[1]) + voc[1]);
    result[1] = dot(P, D) / D.squaredNorm();
    if (result[0] > result[1]) {    
      um2::swap(result[0], result[1]);    
    }
  }    
  return result; 
}

} // namespace um2
