namespace um2
{

//==============================================================================
// LineSegment
//==============================================================================

template <Size D, typename T>
PURE HOSTDEV constexpr auto
pointClosestTo(LineSegment<D, T> const & l, Point<D, T> const & p) noexcept -> T
{
  // From Real-Time Collision Detection, Christer Ericson, 2005
  // Given segment ab and point c, computes closest point d on ab.
  // Returns t for the position of d, d(r) = a + r*(b - a)
  Vec<D, T> const ab = l[1] - l[0];
  // Project c onto ab, computing parameterized position d(r) = a + r*(b − a)
  T r = (p - l[0]).dot(ab) / ab.squaredNorm();
  // If outside segment, clamp r (and therefore d) to the closest endpoint
  if (r < 0) {
    r = 0;
  }
  if (r > 1) {
    r = 1;
  }
  return r;
}

//==============================================================================
// QuadraticSegment
//==============================================================================

// NOLINTBEGIN(readability-identifier-naming) justification: Mathematical notation
template <Size D, typename T>
PURE HOSTDEV constexpr auto
pointClosestTo(QuadraticSegment<D, T> const & q, Point<D, T> const & p) noexcept -> T
{

  // We want to use the complex funcstions in the std or cuda::std namespace
  // depending on if we're compiling for the host or device
  // NOLINTNEXTLINE(google-build-using-namespace) justified
  using namespace std;

  // Note the 1-based indexing in this section
  //
  // The interpolation function of the quadratic segment is
  // Q(r) = C + rB + r²A,    
  // where
  // C = P₁    
  // B = 3V₁₃ + V₂₃    = -3q[1] -  q[2] + 4q[3]
  // A = -2(V₁₃ + V₂₃) =  2q[1] + 2q[2] - 4q[3]
  // V₁₃ = q[3] - q[1]
  // V₂₃ = q[3] - q[2]
  //
  // We wish to find r which minimizes ‖P - Q(r)‖.
  // This r also minimizes ‖P - Q(r)‖².
  // It can be shown that this is equivalent to finding the minimum of the
  // quartic function
  // ‖P - Q(r)‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
  // Let W = P - P₁ = P - C
  // a₄ = A ⋅ A
  // a₃ = 2(A ⋅ B) 
  // a₂ = -2(A ⋅ W) + (B ⋅ B)
  // a₁ = -2(B ⋅ W)
  // a₀ = W ⋅ W
  //
  // The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
  // a = 2(A ⋅ A)
  // b = 3(A ⋅ B)
  // c = (B ⋅ B) - 2(A ⋅W)
  // d = -(B ⋅ W)
  // Note we factored out a 2 
  //
  // We can then use Lagrange's method is used to find the roots.
  // (https://en.wikipedia.org/wiki/Cubic_equation#Lagrange's_method)
  Vec<D, T> const v13 = q[2] - q[0];
  Vec<D, T> const v23 = q[2] - q[1];
  Vec<D, T> const A = -2 * (v13 + v23);
  T const a = 2 * squaredNorm(A); 
  // 0 ≤ a, since a = 2(A ⋅ A)  = 2 ‖A‖², and 0 ≤ ‖A‖²
  // A = 4(midpoint of line - p3) -> a = 32 ‖midpoint of line - p3‖²
  // if a is small, then the segment is almost a straight line, and we should use the
  // line segment method
  if (a < 32 * epsilonDistanceSquared<T>()) {
    Vec<D, T> const ab = q[1] - q[0];
    T r = (p - q[0]).dot(ab) / ab.squaredNorm();
    if (r < 0) {
      r = 0;
    }
    if (r > 1) {
      r = 1;
    }
    return r;
  }
  Vec<D, T> const B = 3 * v13 + v23;
  T const b = 3 * dot(A, B); 
  Vec<D, T> const W = p - q[0];
  T const c = squaredNorm(B) - 2 * dot(A, W);
  T const d = -dot(B, W);

  // Lagrange's method
  // Compute the elementary symmetric functions
  T const e1 = -b / a; // Note for later s0 = e1
  T const e2 = c / a;
  T const e3 = -d / a;
  // Compute the symmetric functions
  T const P = e1 * e1 - 3 * e2;
  T const S = 2 * e1 * e1 * e1 - 9 * e1 * e2 + 27 * e3;
  // We solve z^2 - Sz + P^3 = 0
  T const disc = S * S - 4 * P * P * P; 
  T const eps = static_cast<T>(1e-7);
//  assert(um2::abs(disc) > eps); // 0 single or double root
//  if (0 < disc) { // One real root
//    T const s1 = um2::cbrt((S + um2::sqrt(disc)) / 2);
//    T const s2 = (um2::abs(s1) < eps) ? 0 : P / s1;
//    // Using s0 = e1
//    return (e1 + s1 + s2) / 3;
//  }
  // A complex cbrt
  T constexpr half = static_cast<T>(0.5);
  T constexpr third = static_cast<T>(1) / 3;
  complex<T> const s1 = exp(log((S + sqrt(static_cast<complex<T>>(disc))) * half) * third);
  complex<T> const s2 = (abs(s1) < eps) ? 0 : P / s1;
  // zeta1 = (-1/2, sqrt(3)/2)
  complex<T> const zeta1(static_cast<T>(-0.5), um2::sqrt(static_cast<T>(3)) / 2);
  complex<T> const zeta2(conj(zeta1));

  // Find the real root that minimizes the distance to p
  T r = 0; 
  T dist = p.squaredDistanceTo(q[0]);
  if (p.squaredDistanceTo(q[1]) < dist) {
    r = 1;
    dist = p.squaredDistanceTo(q[1]);
  }

  Vec3<T> const rr((e1 + real(s1 + s2)) / 3,
                   (e1 + real(zeta2 * s1 + zeta1 * s2)) / 3,
                   (e1 + real(zeta1 * s1 + zeta2 * s2)) / 3);
  for (Size i = 0; i < 3; ++i) {
    T const rc = rr[i];
    if (0 <= rc && rc <= 1) {
      T const dc = p.squaredDistanceTo(q(rc));
      if (dc < dist) {
        r = rc;
        dist = dc;
      }
    }
  }
  return r; 
}
// NOLINTEND(readability-identifier-naming)

} // namespace um2
