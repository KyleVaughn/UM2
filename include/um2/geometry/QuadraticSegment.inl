namespace um2
{

// -------------------------------------------------------------------
// Accessors
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Size i) noexcept -> Point<D, T> &
{
  return vertices[i];
}

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator[](Size i) const noexcept -> Point<D, T> const &
{
  return vertices[i];
}

// -------------------------------------------------------------------
// Constructors
// -------------------------------------------------------------------

template <Size D, typename T>
HOSTDEV constexpr QuadraticSegment<D, T>::Polytope(Point<D, T> const & p0,
                                                   Point<D, T> const & p1,
                                                   Point<D, T> const & p2) noexcept
{
  vertices[0] = p0;
  vertices[1] = p1;
  vertices[2] = p2;
}

// -------------------------------------------------------------------
// Interpolation
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::operator()(R const r) const noexcept -> Point<D, T>
{
  // (2 * r - 1) * (r - 1) * v0 +
  // (2 * r - 1) *  r      * v1 +
  // -4 * r      * (r - 1) * v2
  T const rr = static_cast<T>(r);
  // If an add or sub is 4 cycles and mul is 7, then we can solve for the weights quickly using 
  // the following:
  T const two_rr = 2 * rr;
  T const rr_1 = rr - 1;
  T const x = two_rr * rr_1; 

  T const w0 = x - rr_1;
  T const w1 = (two_rr - 1) * rr;      
  T const w2 = -2 * x; 
  Point<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] =
        w0 * this->vertices[0][i] + w1 * this->vertices[1][i] + w2 * this->vertices[2][i];
  }
  return result;
}

// -------------------------------------------------------------------
// jacobian
// -------------------------------------------------------------------

template <Size D, typename T>
template <typename R>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::jacobian(R r) const noexcept -> Vec<D, T>
{
  // (4 * r - 3) * (v0 - v2) + (4 * r - 1) * (v1 - v2)
  T const w0 = 4 * static_cast<T>(r) - 3;
  T const w1 = 4 * static_cast<T>(r) - 1;
  Vec<D, T> result;
  for (Size i = 0; i < D; ++i) {
    result[i] =
        w0 * (this->vertices[0][i] - this->vertices[2][i]) +
        w1 * (this->vertices[1][i] - this->vertices[2][i]);
  }
  return result;
}

//// -------------------------------------------------------------------
//// isLeft
//// -------------------------------------------------------------------
//
// template <Size D, typename T>
// PURE HOSTDEV constexpr auto
// QuadraticSegment<D, T>::isLeft(Point<D, T> const & p) const noexcept -> bool requires(D
// == 2)
//{
//  return areCCW(vertices[0], vertices[1], p);
//}
//
//// -------------------------------------------------------------------
//// length
//// -------------------------------------------------------------------
//
// template <Size D, typename T>
// PURE HOSTDEV constexpr auto
// QuadraticSegment<D, T>::length() const noexcept -> T
//{
//  return vertices[0].distanceTo(vertices[1]);
//}
//
// -------------------------------------------------------------------
// boundingBox
// -------------------------------------------------------------------

template <Size D, typename T>
PURE HOSTDEV constexpr auto
QuadraticSegment<D, T>::boundingBox() const noexcept -> AxisAlignedBox<D, T>
{
  // Find the extrema by finding dx_i/dr = 0
  //  Q(r) = P₁ + r𝘂 + r²𝘃,            
  // where            
  //  𝘂 = 3𝘃₁₃ + 𝘃₂₃    = -3q[1] -  q[2] + 4q[3]     
  //  𝘃 = -2(𝘃₁₃ + 𝘃₂₃) =  2q[1] + 2q[2] - 4q[3]           
  // and            
  // 𝘃₁₃ = q[3] - q[1]            
  // 𝘃₂₃ = q[3] - q[2]
  // Q′(r) = 𝘂 + 2r𝘃,            
  // (r_i,...) = -𝘂 / (2𝘃)    
  // Compare the extrema with the segment's endpoints to find the AABox
  Vec<D, T> v02;
  Vec<D, T> v12;
  for (Size i = 0; i < D; ++i) {
    v02[i] = vertices[2][i] - vertices[0][i];
    v12[i] = vertices[2][i] - vertices[1][i];
  }
  Vec<D, T> u;
  Vec<D, T> v;
  for (Size i = 0; i < D; ++i) {
    u[i] =  3 * v02[i] + v12[i];
    v[i] = -2 * (v02[i] + v12[i]);
  }
  Vec<D, T> r;
  for (Size i = 0; i < D; ++i) {
    // r_i = -𝘂_i / (2𝘃_i)
    r[i] = -u[i] / (2 * v[i]); 
  }
  Point<D, T> stationary; 
  for (Size i = 0; i < D; ++i) {
    stationary[i] = vertices[0][i] + r[i] * (u[i] + r[i] * v[i]); 
  }
  Point<D, T> minima = vertices[0]; 
  minima.min(vertices[1]);
  Point<D, T> maxima = vertices[0];
  maxima.max(vertices[1]);
//  for (Size i = 0; i < D; ++i) {
//    /
//    if
//  }

  return AxisAlignedBox<D, T>{minima, maxima};
}

} // namespace um2
