namespace um2 {

//template <typename T>
//UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
//T & Mat2x2<T>::operator [] (len_t const i, len_t const j) 
//requires (!is_simd_vector<M, T>)
//{
//    UM2_ASSERT(0 <= i && i < M && 0 <= j && j < N);
//    return this->cols[j][i];
//}
//
//template <typename T>
//UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
//T const & Mat2x2<T>::operator [] (len_t const i, len_t const j) const 
//requires (!is_simd_vector<M, T>)
//{
//    UM2_ASSERT(0 <= i && i < M && 0 <= j && j < N);
//    return this->cols[j][i];
//}
//
//template <typename T>
//UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
//T Mat2x2<T>::operator [] (len_t const i, len_t const j) const 
//requires (is_simd_vector<M, T>)
//{
//    UM2_ASSERT(0 <= i && i < M && 0 <= j && j < N);
//    return this->cols[j][i];
//}
//
//// -- Constructors --
//
//// From a list of columns    
//template <typename T>
//template<std::same_as<Vec2<T>> ...Cols>                                     
//requires (sizeof...(Cols) == N)  
//UM2_HOSTDEV constexpr Mat2x2<T>::Mat(Cols... in_cols) : cols{in_cols...} {}
//
//// -- Unary operators --
//
//template <typename T>
//UM2_CONST UM2_HOSTDEV constexpr
//Mat2x2<T> operator - (Mat2x2<T> A) {
//    for (len_t i = 0; i < N; ++i) { A.cols[i] = -A.cols[i]; }
//    return A;
//}
//
//// -- Binary operators --
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Mat2x2<T> operator + (Mat2x2<T> A, Mat2x2<T> const & B) {
//    for (len_t i = 0; i < N; ++i) { A.cols[i] += B.cols[i]; }
//    return A;
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Mat2x2<T> operator - (Mat2x2<T> A, Mat2x2<T> const & B) {
//    for (len_t i = 0; i < N; ++i) { A.cols[i] -= B[i]; }
//    return A;
//}
//
//// -- Scalar operators --
//
//template <typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr
//Mat2x2<T> & operator *= (Mat2x2<T> & A, S const s) {
//    for (len_t i = 0; i < N; ++i) { A.cols[i] *= static_cast<T>(s); }
//    return A;
//}
//
//template <typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_HOSTDEV constexpr
//Mat2x2<T> & operator /= (Mat2x2<T> & A, S const s) {
//    for (len_t i = 0; i < N; ++i) { A.cols[i] /= static_cast<T>(s); }
//    return A;
//}
//
//template <typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_CONST UM2_HOSTDEV constexpr
//Mat2x2<T> operator * (S const s, Mat2x2<T> A) {
//    return A *= static_cast<T>(s);
//}
//
//template <typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_CONST UM2_HOSTDEV constexpr
//Mat2x2<T> operator * (Mat2x2<T> A, S const s) {
//    return A *= static_cast<T>(s);
//}
//
//template <typename T, typename S>
//requires (std::same_as<T, S> || (std::floating_point<T> && std::integral<S>))
//UM2_CONST UM2_HOSTDEV constexpr
//Mat2x2<T> operator / (Mat2x2<T> A, S const s) {
//    return A /= static_cast<T>(s);
//}
//
//// -- Mat2x2 -- 
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Vec2<T> operator * (Mat2x2<T> const & A, Vec2<T> const & v) {
//    return Vec2<T>{A[0, 0] * v[0] + A[0, 1] * v[1],
//                      A[1, 0] * v[0] + A[1, 1] * v[1]};
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Mat2x2<T> operator * (Mat2x2<T> const & A, Mat2x2<T> const & B) {
//    return Mat2x2<T>{A * B[0], A * B[1]};
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//T det(Mat2x2<T> const & A) {
//    return cross(A[0], A[1]);
//}
//
//template <typename T>
//UM2_PURE UM2_HOSTDEV constexpr
//Mat2x2<T> inv(Mat2x2<T> const & A) {
//    T const inv_det = static_cast<T>(1) / det(A);
//    Mat2x2<T> B = Mat2x2<T>{Vec2<T>{ A[1, 1], -A[1, 0]},
//                            Vec2<T>{-A[0, 1],  A[0, 0]}};
//    B *= inv_det;
//    return B;
//}
//
} // namespace um2
