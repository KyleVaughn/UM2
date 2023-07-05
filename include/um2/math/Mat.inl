namespace um2
{

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::col(Size i) noexcept -> typename Mat<M, N, T>::Col &
{
  assert(i < N);
  return cols[i];
}

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::col(Size i) const noexcept -> typename Mat<M, N, T>::Col const &
{
  assert(i < N);
  return cols[i];
}

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::operator()(Size i, Size j) noexcept -> T &
{
  assert(i < M && j < N);
  return cols[j][i];
}

template <Size M, Size N, typename T>
PURE HOSTDEV constexpr auto
Mat<M, N, T>::operator()(Size i, Size j) const noexcept -> T const &
{
  assert(i < M && j < N);
  return cols[j][i];
}
//
//// -- Constructors --
//
//// From a list of columns
// template <Size M, Size N, typename T>
// template<std::same_as<Vec<M, T, Q>> ...Cols>
// requires (sizeof...(Cols) == N)
// HOSTDEV constexpr Mat<M, N, T, Q>::Mat(Cols... in_cols) : cols{in_cols...} {}
//
//// -- IO --
//
// template <Size M, Size N, typename T>
// std::ostream & operator << (std::ostream & os, Mat<M, N, T, Q> const & A) {
//    Size width = 0;
//    for (Size i = 0; i < N; ++i) {
//        for (Size j = 0; j < M; ++j) {
//          width = std::max(width, std::to_string(A(i, j)).size());
//        }
//    }
//    os << "[";
//    for (Size i = 0; i < N; ++i) {
//        if (i != 0) os << " ";
//        for (Size j = 0; j < M; ++j) {
//            Size this_width = std::to_string(A(i, j)).size();
//            Size padding = width - this_width;
//            os << std::string(padding, ' ') << A(i, j);
//            if (j < M - 1) os << ", ";
//        }
//        if (i < N - 1) os << std::endl;
//    }
//    os << "]";
//    return os;
//}
//
//// -- Unary operators --
//
// template <Size M, Size N, typename T>
// CONST HOSTDEV constexpr
// Mat<M, N, T, Q> operator - (Mat<M, N, T, Q> A) {
//    for (Size i = 0; i < N; ++i) { A.cols[i] = -A.cols[i]; }
//    return A;
//}
//
//// -- Binary operators --
//
// template <Size M, Size N, typename T>
// PURE HOSTDEV constexpr
// Mat<M, N, T, Q> operator + (Mat<M, N, T, Q> A, Mat<M, N, T, Q> const & B) {
//    for (Size i = 0; i < N; ++i) { A.cols[i] += B.cols[i]; }
//    return A;
//}
//
// template <Size M, Size N, typename T>
// PURE HOSTDEV constexpr
// Mat<M, N, T, Q> operator - (Mat<M, N, T, Q> A, Mat<M, N, T, Q> const & B) {
//    for (Size i = 0; i < N; ++i) { A.cols[i] -= B[i]; }
//    return A;
//}
//
//// -- Scalar operators --
//
// template <Size M, Size N, typename T, typename S>
// requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
// HOSTDEV constexpr
// Mat<M, N, T, Q> & operator *= (Mat<M, N, T, Q> & A, S const s) {
//    for (Size i = 0; i < N; ++i) { A.cols[i] *= static_cast<T>(s); }
//    return A;
//}
//
// template <Size M, Size N, typename T, typename S>
// requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
// HOSTDEV constexpr
// Mat<M, N, T, Q> & operator /= (Mat<M, N, T, Q> & A, S const s) {
//    for (Size i = 0; i < N; ++i) { A.cols[i] /= static_cast<T>(s); }
//    return A;
//}
//
// template <Size M, Size N, typename T, typename S>
// requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
// CONST HOSTDEV constexpr
// Mat<M, N, T, Q> operator * (S const s, Mat<M, N, T, Q> A) {
//    return A *= static_cast<T>(s);
//}
//
// template <Size M, Size N, typename T, typename S>
// requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
// CONST HOSTDEV constexpr
// Mat<M, N, T, Q> operator * (Mat<M, N, T, Q> A, S const s) {
//    return A *= static_cast<T>(s);
//}
//
// template <Size M, Size N, typename T, typename S>
// requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
// CONST HOSTDEV constexpr
// Mat<M, N, T, Q> operator / (Mat<M, N, T, Q> A, S const s) {
//    return A /= static_cast<T>(s);
//}
//
//// -- Mat2x2 --
//
// template <typename T>
// PURE HOSTDEV constexpr
// Vec2<T, Q> operator * (Mat2x2<T, Q> const & A, Vec2<T, Q> const & v) {
//    return Vec2<T, Q>{A[0, 0] * v[0] + A[0, 1] * v[1],
//                      A[1, 0] * v[0] + A[1, 1] * v[1]};
//}
//
// template <typename T>
// PURE HOSTDEV constexpr
// Mat2x2<T, Q> operator * (Mat2x2<T, Q> const & A, Mat2x2<T, Q> const & B) {
//    return Mat2x2<T>{A * B[0], A * B[1]};
//}
//
// template <typename T>
// PURE HOSTDEV constexpr
// T det(Mat2x2<T, Q> const & A) {
//    return cross(A[0], A[1]);
//}
//
// template <typename T>
// PURE HOSTDEV constexpr
// Mat2x2<T> inv(Mat2x2<T> const & A) {
//    T const inv_det = static_cast<T>(1) / det(A);
//    Mat2x2<T> B = Mat2x2<T>{Vec2<T>{ A[1, 1], -A[1, 0]},
//                            Vec2<T>{-A[0, 1],  A[0, 0]}};
//    B *= inv_det;
//    return B;
//}

} // namespace um2
